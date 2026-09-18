#!/usr/bin/env python
"""Offline self-test of the whole pipeline (no network, no GPU needed, ~1 minute).

Builds a synthetic corpus, a tiny randomly-initialised Qwen2-architecture model and a tiny
embedding model under selftest_work/, checks the data-parsing helpers of 01_pull_data.py,
then runs 02_generate.py -> 03_score.py -> 04_digest.py on them with selftest_work/config.yaml.

The generated text is gibberish (random weights); the point is that every code path runs.
"""

import io
import json
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "selftest_work"
sys.path.insert(0, str(HERE))

import importlib  # noqa: E402

pull = importlib.import_module("01_pull_data")
from common import read_jsonl, write_jsonl_atomic  # noqa: E402

WORDS = ("servo loop gain sensor feedback actuator valve pressure encoder shaft thermocouple bridge amplifier "
         "controller setpoint error integral derivative sampling interval measurement instrument transducer "
         "damping oscillation filter signal noise calibration reference voltage current resistor coil flow "
         "temperature position velocity torque motor stepper pulse counter timer register bus interface").split()


def rand_text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def make_corpus(rng):
    failures = []
    for i in range(40):
        failures.append({
            "id": f"hupd:{10000000 + i}", "source": "hupd", "application_number": str(10000000 + i),
            "title": f"Apparatus for {rand_text(rng, 4)}", "abstract": rand_text(rng, 60).capitalize() + ".",
            "first_claim": "1. An apparatus comprising " + rand_text(rng, 40) + ".",
            "background": rand_text(rng, 30), "cpc": "G05B11/42", "ipc": "G05B11/42",
            "decision": "REJECTED", "filing_year": 2004 + i % 15,
        })
    retro = []
    for i in range(30):
        retro.append({
            "id": f"utzoo:{i:012x}", "source": "utzoo", "group": "net.electronics",
            "subject": f"Re: {rand_text(rng, 3)}", "date": "12 Mar 1986", "text": rand_text(rng, 150) + ".",
            "word_count": 151, "keyword_hits": 5,
        })
    return failures, retro


def make_tiny_qwen(model_dir, texts):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    import torch

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1500, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    fast.save_pretrained(model_dir)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(model_dir)


def make_tiny_embedder(model_dir, texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    import torch

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", model_max_length=256)
    fast.save_pretrained(model_dir)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(fast), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256)
    BertModel(cfg).save_pretrained(model_dir)


def test_pull_helpers():
    claims = ("1. A control apparatus comprising a sensor; and a controller coupled to the sensor.\n\n"
              "2. The apparatus of claim 1, wherein the sensor is a thermocouple.\n\n3. The apparatus of claim 2.")
    c1 = pull.first_independent_claim(claims)
    assert c1.startswith("A control apparatus") and "thermocouple" not in c1, c1
    one_line = "1. A widget comprising a lever. 2. The widget of claim 1 wherein the lever is steel."
    assert pull.first_independent_claim(one_line) == "A widget comprising a lever.", pull.first_independent_claim(one_line)

    article = ("Path: utzoo!decvax!foo\nFrom: henry@utzoo.UUCP (Henry Spencer)\nNewsgroups: net.electronics,net.micro\n"
               "Subject: Re: PID loop tuning\nDate: 12 Mar 86 04:20:11 GMT\n\n"
               "In article <123@bar.UUCP> joe@bar.UUCP writes:\n> how do I tune this thing?\n\n"
               "Set the integral term to zero first. Then raise the proportional gain until the loop oscillates.\n"
               "Back it off by half. The sensor lag matters more than the actuator here.\n\n-- \nHenry Spencer\n"
               "{allegra,ihnp4,decvax}!utzoo!henry\n")
    headers, body = pull.split_usenet_headers(article)
    assert headers["newsgroups"] == "net.electronics,net.micro" and headers["subject"] == "Re: PID loop tuning"
    cleaned = pull.clean_usenet_body(body)
    assert cleaned.startswith("Set the integral") and "writes:" not in cleaned and "Henry" not in cleaned, cleaned
    assert pull.groups_of({}, {}, headers) == ["net.electronics", "net.micro"]

    # nested tar streaming (all-years.tar containing 2005.tar.gz containing json files)
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as tf:
        for app in ("111", "222"):
            data = json.dumps({"application_number": app, "title": "T" + app, "abstract": "A" + app,
                               "claims": claims, "background": "B", "main_cpc_label": "G05B1/00"}).encode()
            ti = tarfile.TarInfo(f"2005/{app}.json")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    outer = io.BytesIO()
    with tarfile.open(fileobj=outer, mode="w") as tf:
        ti = tarfile.TarInfo("data/2005.tar.gz")
        ti.size = len(inner.getvalue())
        tf.addfile(ti, io.BytesIO(inner.getvalue()))
    names = [n for n, _ in pull.iter_tar_files(io.BytesIO(outer.getvalue()))]
    assert names == ["2005/111.json", "2005/222.json"], names
    rec = pull.hupd_record(json.loads(dict(pull.iter_tar_files(io.BytesIO(outer.getvalue())))["2005/111.json"]),
                           {"application_number": "111", "filing_year": 2005, "background_max_chars": 100,
                            "decision": "REJECTED", "cpc": "G05B1/00"})
    assert rec["id"] == "hupd:111" and rec["first_claim"].startswith("A control apparatus"), rec
    print("[selftest] 01_pull_data helpers OK")


def run(cmd):
    print("[selftest] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(HERE))


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()
    rng = random.Random(0)
    test_pull_helpers()

    failures, retro = make_corpus(rng)
    (WORK / "data").mkdir()
    write_jsonl_atomic(WORK / "data" / "failures.jsonl", failures)
    write_jsonl_atomic(WORK / "data" / "retro.jsonl", retro)
    texts = [f["abstract"] for f in failures] + [r["text"] for r in retro]
    make_tiny_qwen(WORK / "tiny-qwen2-base", texts)
    make_tiny_embedder(WORK / "tiny-embedder", texts)

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg["paths"] = {k: str(WORK / v) for k, v in cfg["paths"].items()}
    cfg["model"]["path"] = str(WORK / "tiny-qwen2-base")
    cfg["generation"].update(n_samples=24, backend="transformers", batch_size=8, max_new_tokens=40)
    cfg["scoring"]["embedding_model"] = str(WORK / "tiny-embedder")
    cfg["digest"]["top_n"] = 5
    (WORK / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    py = sys.executable
    date = "2000-01-01"
    run([py, "02_generate.py", "--config", str(WORK / "config.yaml"), "--date", date, "--chunk", "16"])
    rows = read_jsonl(WORK / "runs" / f"{date}.jsonl")
    assert len(rows) == 24 and all(len(r["source_ids"]) == 4 for r in rows)
    # idempotence: a second run adds nothing
    run([py, "02_generate.py", "--config", str(WORK / "config.yaml"), "--date", date])
    assert len(read_jsonl(WORK / "runs" / f"{date}.jsonl")) == 24
    # determinism of source selection
    from importlib import import_module
    gen = import_module("02_generate")
    from common import load_config
    c = load_config(WORK / "config.yaml")
    s = gen.build_sample(date, 3, c, sorted(failures, key=lambda r: r["id"]), sorted(retro, key=lambda r: r["id"]),
                         lambda t: len(t.split()))
    assert s["source_ids"] == rows[3]["source_ids"] and s["seed"] == rows[3]["seed"]
    assert s["prompt"].endswith(c["generation"]["suffix"])

    run([py, "03_score.py", "--config", str(WORK / "config.yaml"), "--date", date])
    scores = read_jsonl(WORK / "scores" / f"{date}.jsonl")
    kept = [r for r in scores if r["band"] == "keep"]
    valid = [r for r in scores if r["band"] != "drop_empty"]
    assert len(kept) == len(valid) - 2 * round(0.1 * len(valid)), (len(kept), len(valid))
    assert [r["rank"] for r in kept] == list(range(1, len(kept) + 1))
    assert all(kept[i]["novelty"] >= kept[i + 1]["novelty"] for i in range(len(kept) - 1))
    run([py, "03_score.py", "--config", str(WORK / "config.yaml"), "--date", date])  # cache hit path

    run([py, "04_digest.py", "--config", str(WORK / "config.yaml"), "--date", date])
    digest = (WORK / "digests" / f"{date}.md").read_text()
    by_id = {r["id"]: r for r in rows}
    for r in kept[:5]:
        assert by_id[r["id"]]["output"] in digest, "digest must contain the verbatim output"
        assert f"seed: {r['seed']}" in digest
    print("[selftest] ALL OK")
    print(digest[:1500])


if __name__ == "__main__":
    main()
