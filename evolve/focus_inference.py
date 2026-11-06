"""Read-only text continuation from the latest accepted local focus checkpoint."""
import hashlib
import io
import math
from pathlib import Path
from threading import Lock
import time

from .focus_training import FORMAT, CONTEXT, _network, _validate_model_config, parameter_count

NOTE = "Experimental text continuation. This model has not been trained or evaluated to provide reliable historical answers."


def active_model(store, profile="saudi-history-ar"):
    record = store.get("focus_train_latest_" + profile) or {}
    if record.get("accepted") is not True or record.get("format") not in (FORMAT, "berhot_owned_byte_gru_v1"):
        return None
    root = (store.root / "models" / "focus" / profile).resolve()
    path = Path(record.get("checkpoint", "")).resolve()
    digest = record.get("sha256", "")
    if (not path.is_relative_to(root) or not path.is_file() or
            path.suffix != (".bin" if record.get('format')=='berhot_owned_byte_gru_v1' else ".pt") or len(digest) != 64 or
            path.stat().st_size > (160_000_000 if record.get('format')=='berhot_owned_byte_gru_v1' else 64_000_000)):
        raise ValueError("Accepted checkpoint is missing or outside the local model folder")
    config = _validate_model_config(record.get("model_config") or {})
    return record, path, config


def brain_status(store, profile="saudi-history-ar"):
    from .focus_control import worker_running
    run = store.get("focus_run_" + profile) or {}
    result = {"status": "no_accepted_checkpoint", "model": {}, "note": NOTE,
              "capabilities": {"instruction_tuned": False, "history_qa_evaluated": False,
                               "historical_answers_verified": False,
                               "acceptance_measures": "next_UTF8_byte_prediction_loss"},
              "activity": {"phase": run.get("current_phase", "not_started"),
                           "worker_running": worker_running(store, profile)}}
    result['full_pass']=store.get('focus_full_pass_progress')
    try:
        active = active_model(store, profile)
    except ValueError:
        result.update(status="checkpoint_unavailable", error="The accepted checkpoint cannot be loaded from its saved path.")
        return result
    if active:
        record, _, config = active
        result.update(status="ready", model={"checkpoint_sha256": record["sha256"],
            "parameters": parameter_count(config), "accepted_training_steps": record.get("steps_total", 0),
            "created_at_utc": record.get("created_at_utc"), "model_type": "byte_gru", "profile": profile,
            "training_context_bytes": record.get("context_bytes", CONTEXT),
            "training_objective": "next_byte_prediction",
            "validation_loss": record.get("validation_after")})
        result['model']['training_engine']=record.get('training_engine','legacy_pytorch')
        result['model']['external_ml_runtime_required']=record.get('format')!='berhot_owned_byte_gru_v1'
        result['model']['instruction_tuned']=bool(record.get('instruction_tuned'))
        result['model']['supervised_qa_steps']=int(record.get('supervised_qa_steps_total',0))
        result['capabilities']['instruction_tuned']=bool(record.get('instruction_tuned'))
        result['capabilities']['qa_answer_loss_evaluated']=record.get('qa_validation_loss_after') is not None
        result['qa_dataset']=store.get('focus_supervised_qa_latest')
        cursor=store.get('owned_training_cursor_'+profile) or {}
        if cursor.get('dataset_revision_sha256')==record.get('dataset_revision_sha256'):
            offsets=cursor.get('document_offsets')
            targets=int(record.get('unique_approved_training_bytes',0))-int((record.get('documents_by_split') or {}).get('train',0))
            result['training_progress']={
                'completed_full_passes':int(cursor.get('epoch',0))+int(offsets is not None and targets>0 and sum(offsets)>=targets),
                'current_pass_fraction':sum(offsets)/targets if offsets is not None and targets>0 else None,
                'documents_touched_this_pass':cursor.get('documents_touched'),
                'training_documents':(record.get('documents_by_split') or {}).get('train'),
                'optimizer_attempt_bytes':cursor.get('attempted_training_bytes',0),
                'meaning':'Training attempts and coverage; not verified facts or accepted learned bytes.'}
    return result


def _allowed_bytes(prefix, remaining):
    """Constrain sampling to printable, valid UTF-8, never rewrite model text."""
    if prefix:
        lead = prefix[0]
        low, high = 0x80, 0xBF
        if len(prefix) == 1:
            if lead == 0xE0: low = 0xA0
            if lead == 0xED: high = 0x9F
            if lead == 0xF0: low = 0x90
            if lead == 0xF4: high = 0x8F
        return list(range(low, high + 1))
    allowed = [9, 10] + list(range(32, 127))
    if remaining >= 2: allowed += list(range(0xC2, 0xE0))
    if remaining >= 3: allowed += list(range(0xE0, 0xF0))
    if remaining >= 4: allowed += list(range(0xF0, 0xF5))
    return allowed


class FocusGenerator:
    def __init__(self):
        self.lock = Lock()
        self.cached_key = None
        self.cached = None

    def generate(self, store, prompt, *, max_new_bytes=256, temperature=0.8,
                 top_k=40, seed=42, profile="saudi-history-ar"):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1000:
            raise ValueError("Prompt must contain 1..1000 characters")
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= 1024:
            raise ValueError("max_new_bytes must be 1..1024")
        if type(top_k) is not int or not 1 <= top_k <= 256:
            raise ValueError("top_k must be 1..256")
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError("seed must be 0..2147483647")
        if (type(temperature) not in (int, float) or not math.isfinite(temperature)
                or not 0 <= temperature <= 1.5):
            raise ValueError("temperature must be 0..1.5")
        if not self.lock.acquire(blocking=False):
            return {"status": "busy", "answer": "", "note": "A model generation request is already running."}
        try:
            if (store.get('focus_train_latest_'+profile) or {}).get('format')=='berhot_owned_byte_gru_v1':
                from .owned_learning import generate_owned
                return generate_owned(self,store,prompt,max_new_bytes,temperature,top_k,seed)
            active = active_model(store, profile)
            if active is None:
                return {"status": "no_accepted_checkpoint", "answer": "", "note": "No accepted Arabic model is available yet."}
            record, path, config = active
            stat = path.stat()
            cache_key = (str(path), record["sha256"], stat.st_size, stat.st_mtime_ns)
            if cache_key != self.cached_key:
                blob = path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                    raise ValueError("Accepted checkpoint SHA-256 verification failed")
                torch, model = _network(config)
                torch.set_num_threads(min(4, torch.get_num_threads()))
                saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
                if saved.get("format") != FORMAT or _validate_model_config(saved.get("model_config") or {}) != config:
                    raise ValueError("Accepted checkpoint format or architecture mismatch")
                model.load_state_dict(saved["weights"], strict=True)
                model.eval()
                self.cached, self.cached_key = (torch, model), cache_key
            torch, model = self.cached
            generator = torch.Generator(device="cpu").manual_seed(seed)
            # This model learned short continuations; no instruction wrapper,
            # retrieved text, remote model or fake conversational memory.
            context = prompt.encode("utf-8")[-512:]
            output, prefix = bytearray(), bytearray()
            started = time.monotonic()
            with torch.inference_mode():
                ids = torch.tensor([list(context)], dtype=torch.long)
                states, hidden = model.gru(model.embedding(ids))
                logits = model.head(states[:, -1, :])[0]
                for index in range(max_new_bytes):
                    allowed = _allowed_bytes(prefix, max_new_bytes - index)
                    choices = torch.tensor(allowed, dtype=torch.long)
                    scores = logits[choices]
                    if not torch.isfinite(scores).all():
                        raise ValueError("Checkpoint produced non-finite probabilities")
                    if temperature <= 0.000001:
                        selected = int(scores.argmax())
                    else:
                        values, indices = torch.topk(scores / temperature, min(top_k, len(allowed)))
                        selected = int(indices[torch.multinomial(torch.softmax(values, dim=0), 1, generator=generator)])
                    value = allowed[selected]
                    output.append(value)
                    if prefix:
                        prefix.append(value)
                        length = 2 if prefix[0] < 0xE0 else 3 if prefix[0] < 0xF0 else 4
                        if len(prefix) == length: prefix.clear()
                    elif value >= 0xC2:
                        prefix.append(value)
                    ids = torch.tensor([[value]], dtype=torch.long)
                    states, hidden = model.gru(model.embedding(ids), hidden)
                    logits = model.head(states[:, -1, :])[0]
            return {"status": "generated", "mode": "focus", "answer": output.decode("utf-8"),
                    "training_engine": "pytorch", "external_model_api_used": False,
                    "pretrained_weights_used": False,
                    "model_generated": True, "checkpoint_sha256": record["sha256"],
                    "steps_total": record.get("steps_total", 0), "trainable_parameters": parameter_count(config),
                    "sources": [], "training_updated": False, "note": NOTE,
                    "generation": {"generated_bytes": len(output), "prompt_bytes_used": len(context),
                                   "elapsed_seconds": round(time.monotonic() - started, 3),
                                   "seed": seed, "temperature": temperature, "top_k": top_k,
                                   "utf8_constrained": True, "conversation_memory": False}}
        finally:
            self.lock.release()
