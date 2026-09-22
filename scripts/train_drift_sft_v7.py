#!/usr/bin/env python
"""
VARIANT G -- corpus v2_3_ch3x. ONE VARIABLE CHANGED vs. variant F: the corpus
(v2_2_bq = v2_1 with the "> " contamination stripped, plus the 138 chapter
entries repeated x3: 414 chapter + 564 brief = 978). Hyperparameters identical to F;
padding_free/packing pinned to the values F actually ran with; checkpoints saved.

--- original variant-F header follows ---
VARIANT F -- corpus v2_1. ONE VARIABLE CHANGED vs. variant E: the corpus.

Every hyperparameter below is byte-identical to train_drift_sft_v5.py (variant E).
The only edits are DATASET_PATH and OUTPUT_DIR. Do not "improve" anything here --
the whole point is that the corpus is the sole difference, so the eval
comparison is clean.

  variant E : final_training_corpus_1.json        739 records, 139 long-form
  variant F : final_training_corpus_v2_1_latest.json  702 records, 138 long-form

Pre-flight (preflight_v6_tokens.py, PASSED): max templated sequence = 2176 tokens
against MAX_SEQ_LEN=2560, so no entry is truncated and no entry loses its EOS.

Base model : Qwen3-14B (Unsloth 4-bit)      [HARD REQUIREMENT]
Method     : QLoRA SFT via Unsloth + TRL SFTTrainer
Target     : NVIDIA L4, 23 GB
"""

import json, hashlib

# ---------------------------------------------------------------------------
# Hard requirements (do not change without review)
# ---------------------------------------------------------------------------
MODEL_NAME       = "unsloth/Qwen3-14B-unsloth-bnb-4bit"   # HARD REQ: match backend Qwen3-14B, 4-bit
DATASET_PATH     = "/workspace/final_training_corpus_v2_3_ch3x.json"  # VARIANT G: the one change (v2_2_bq + chapter branch x3)
NUM_TRAIN_EPOCHS = 2                                       # same as E
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # HARD REQ: attention only
#   -> gate_proj / up_proj / down_proj are explicitly DROPPED (HARD REQ)
LORA_DROPOUT     = 0.05                                    # HARD REQ

# The system message. In v2_1 it is ALREADY substituted in every record (the
# {{SYSTEM_ANCHOR}} placeholder is gone), so this is now an equality check
# rather than a substitution. sha256 ed40b81d... -- verified identical to the
# string variant E substituted in, so the trained-on system prompt is unchanged.
SYSTEM_MESSAGE = (
    "Write in the mode of objective physical realism. Describe actions, "
    "environments, and labor with precision.\n"
    "Output is continuous prose. No headers, labels, or formatting."
)
PLACEHOLDER = "{{SYSTEM_ANCHOR}}"

# ---------------------------------------------------------------------------
# Chosen defaults -- ALL IDENTICAL TO VARIANT E
# ---------------------------------------------------------------------------
MAX_SEQ_LEN            = 2560    # same as E; pre-flight max is 2176, so nothing truncates
LORA_R                 = 16
LORA_ALPHA             = 16
LEARNING_RATE          = 2e-4
PER_DEVICE_BATCH_SIZE  = 2
GRAD_ACCUMULATION      = 8       # effective batch = 16
WARMUP_RATIO           = 0.05
WEIGHT_DECAY           = 0.01
LR_SCHEDULER           = "linear"
OPTIMIZER              = "adamw_8bit"
USE_BF16               = True
USE_GRAD_CHECKPOINTING = "unsloth"
SEED                   = 3407
OUTPUT_DIR             = "/workspace/drift_sft_out_v7"  # VARIANT G: separate dir
LOGGING_STEPS          = 1

RUN_TRAINING = True


def build_dataset(tokenizer):
    """Load records, substitute the system placeholder if present, apply the chat template."""
    with open(DATASET_PATH) as f:
        raw = json.load(f)

    substituted = 0
    already_correct = 0
    texts = []
    for rec in raw:
        msgs = []
        for m in rec["messages"]:
            content = m["content"]
            if m["role"] == "system":
                if content.strip() == PLACEHOLDER:
                    content = SYSTEM_MESSAGE
                    substituted += 1
                elif content == SYSTEM_MESSAGE:
                    already_correct += 1
            msgs.append({"role": m["role"], "content": content})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    from datasets import Dataset
    ds = Dataset.from_dict({"text": texts})
    return ds, substituted, already_correct, raw


def main():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto (bf16 on L4)
        load_in_4bit=True,   # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,  # attention only
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRAD_CHECKPOINTING,
        random_state=SEED,
    )

    dataset, substituted, already_correct, raw = build_dataset(tokenizer)

    # --- Proof: every record carries exactly the intended system prompt, and no
    #     anchor backstory leaked into it. Checked against the SYSTEM MESSAGE of
    #     every record, not just record 0 -- record 0's prose is not the subject.
    sys_msgs = {m["content"] for r in raw for m in r["messages"] if m["role"] == "system"}
    assert len(sys_msgs) == 1, f"{len(sys_msgs)} distinct system prompts, expected 1"
    only_sys = sys_msgs.pop()
    assert PLACEHOLDER not in only_sys, "Placeholder still present!"
    assert only_sys == SYSTEM_MESSAGE, "System prompt differs from variant E's!"
    for banned in ("SETTING:", "CHARACTERS:", "CONDITIONS:", "Ljuder", "Karl", "Kristina"):
        assert banned not in only_sys, f"Unexpected anchor content leaked: {banned}"
    sys_sha = hashlib.sha256(only_sys.encode()).hexdigest()

    sample = dataset[0]["text"]
    assert PLACEHOLDER not in sample

    n_examples = len(dataset)
    eff_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUMULATION
    est_steps = (n_examples * NUM_TRAIN_EPOCHS + eff_batch - 1) // eff_batch

    # --- Pre-flight, re-asserted here so training cannot start on a corpus that
    #     would be truncated. This is the check the run is gated on.
    tlens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in dataset["text"]]
    over = [(raw[i].get("id"), tlens[i]) for i in range(len(tlens)) if tlens[i] > MAX_SEQ_LEN]
    assert not over, f"TRUNCATION: {len(over)} entries exceed {MAX_SEQ_LEN}: {over[:5]}"

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
        padding_free=True,      # OPEN 10: pinned. This is what variant F actually ran with (unsloth auto-enabled it).
        packing=False,          # pinned; F ran with False
        save_strategy="steps",  # charter: intermediate checkpoints (half-epoch)
        save_steps=31,
        save_total_limit=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n" + "=" * 70)
    print("CONFIG SUMMARY -- VARIANT G (corpus v2_3_ch3x)")
    print("=" * 70)
    print(f"Model (HARD REQ)            : {MODEL_NAME}")
    print(f"Dataset (THE ONE CHANGE)   : {DATASET_PATH}")
    print(f"Example count              : {n_examples}")
    print(f"System prompt sha256       : {sys_sha}")
    print(f"  placeholder substituted  : {substituted}")
    print(f"  already correct verbatim : {already_correct}")
    print(f"Max templated tokens       : {max(tlens)}  (limit {MAX_SEQ_LEN}, "
          f"headroom {MAX_SEQ_LEN - max(tlens)})")
    print(f"  entries truncated        : {len(over)}")
    print(f"Num epochs                 : {NUM_TRAIN_EPOCHS}")
    print(f"LoRA target_modules (HARD) : {LORA_TARGET_MODULES}")
    print(f"  dropped (HARD REQ)       : gate_proj, up_proj, down_proj")
    print(f"lora_dropout (HARD REQ)    : {LORA_DROPOUT}")
    print(f"lora_r                     : {LORA_R}")
    print(f"lora_alpha                 : {LORA_ALPHA}")
    print(f"learning_rate              : {LEARNING_RATE}")
    print(f"per_device_batch           : {PER_DEVICE_BATCH_SIZE}")
    print(f"grad_accum                 : {GRAD_ACCUMULATION}")
    print(f"effective_batch            : {eff_batch}")
    print(f"max_seq_len                : {MAX_SEQ_LEN}")
    print(f"warmup_ratio               : {WARMUP_RATIO}")
    print(f"weight_decay               : {WEIGHT_DECAY}")
    print(f"lr_scheduler               : {LR_SCHEDULER}")
    print(f"optimizer                  : {OPTIMIZER}")
    print(f"bf16                       : {USE_BF16}")
    print(f"grad_checkpointing         : {USE_GRAD_CHECKPOINTING}")
    print(f"seed                       : {SEED}")
    print(f"Estimated steps            : {est_steps}  (= {n_examples} x {NUM_TRAIN_EPOCHS} / {eff_batch})")
    print("=" * 70)
    print("\n----- FORMATTED EXAMPLE [0] -----")
    print(sample[:1200])
    print("----- END EXAMPLE -----", flush=True)

    if RUN_TRAINING:
        import time
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        train_result = trainer.train()
        wall = time.time() - t0

        adapter_dir = OUTPUT_DIR + "/adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        m = train_result.metrics
        print("\n" + "#" * 70)
        print("TRAINING COMPLETE -- VARIANT F")
        print("#" * 70)
        print(f"final_loss         : {m.get('train_loss')}")
        print(f"total_steps_run    : {train_result.global_step}")
        print(f"wall_clock_seconds : {wall:.1f}  ({wall/60:.2f} min)")
        print(f"peak_gpu_mem_GB    : {peak_gb:.2f}")
        print(f"adapter_saved_to   : {adapter_dir}")
        print(f"full_metrics       : {m}")
        print("#" * 70)
    else:
        print("\nRUN_TRAINING is False -> NOT calling trainer.train().")


if __name__ == "__main__":
    main()
