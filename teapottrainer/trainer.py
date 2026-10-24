import hashlib
import json
import os
import sys
import time
import torch
import math
import random
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from datasets.utils.logging import disable_progress_bar as disable_ds_progress
from huggingface_hub.utils import disable_progress_bars as disable_hub_progress
disable_ds_progress()
disable_hub_progress()

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    LlamaConfig, LlamaForCausalLM,
    MistralConfig, MistralForCausalLM,
    GPT2Config, GPT2LMHeadModel,
    GemmaConfig, GemmaForCausalLM
)

from .config import TrainingConfig
from .datasets import DatasetRegistry, DatasetSourceConfig


class TeapotTrainer:
    """Advanced Modular Trainer supporting sequential pretrain -> SFT phases, custom loss masking, and full evaluation suites."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.m_cfg = config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_short = self.config.tokenizer.split("/")[-1]
        
        try:
            safe_dict = self.config.model_dump(exclude_defaults=True, mode='json')
            run_hash = hashlib.md5(json.dumps(safe_dict, default=str).encode()).hexdigest()[:6]
        except Exception:
            run_hash = hashlib.md5(repr(self.config).encode()).hexdigest()[:6]
        
        self.output_dir = f"./runs/teapot-{tokenizer_short}-{run_hash}"
        
        tb_log_path = self.config.logging_dir or os.path.join(self.output_dir, "runs")
        self.writer = SummaryWriter(log_dir=tb_log_path)

        # Canned starting prompts for qualitative generation checks
        self.canned_seeds = [
            "The universe is",
            "Once upon a time",
            "User: Hello, how can you help me today?\nAssistant:",
            "Python is a programming language that"
        ]

    def _build_model(self, vocab_size: int):
        m_type = self.m_cfg.model_type.lower()

        if m_type == "custom":
            if not self.m_cfg.custom_config_cls or not self.m_cfg.custom_model_cls:
                raise ValueError("When model_type='custom', you must provide custom_config_cls and custom_model_cls.")
            
            cfg = self.m_cfg.custom_config_cls(
                vocab_size=vocab_size,
                hidden_size=self.m_cfg.hidden_size,
                num_hidden_layers=self.m_cfg.layers,
                num_attention_heads=self.m_cfg.heads,
                intermediate_size=self.m_cfg.intermediate,
                max_position_embeddings=self.m_cfg.seq_len,
            )
            return self.m_cfg.custom_model_cls(cfg)

        elif m_type == "llama":
            cfg = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=self.m_cfg.hidden_size,
                intermediate_size=self.m_cfg.intermediate,
                num_hidden_layers=self.m_cfg.layers,
                num_attention_heads=self.m_cfg.heads,
                num_key_value_heads=self.m_cfg.kv_heads,
                max_position_embeddings=self.m_cfg.seq_len,
                rms_norm_eps=1e-5,
            )
            return LlamaForCausalLM(cfg)

        elif m_type == "gpt2":
            cfg = GPT2Config(
                vocab_size=vocab_size,
                n_embd=self.m_cfg.hidden_size,
                n_layer=self.m_cfg.layers,
                n_head=self.m_cfg.heads,
                n_positions=self.m_cfg.seq_len,
                n_ctx=self.m_cfg.seq_len,
            )
            return GPT2LMHeadModel(cfg)

        else:
            raise ValueError(f"Unsupported model_type: '{m_type}'")

    def _print_model_summary(self, model):
        """Analyzes and prints detailed parameter breakdowns by component."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        breakdown = {}
        for name, param in model.named_parameters():
            num = param.numel()
            if "embed_tokens" in name or "wte" in name or "word_embeddings" in name:
                category = "Embeddings"
            elif "lm_head" in name or "output" in name:
                category = "LM Head / Output"
            elif "self_attn" in name or "attn" in name:
                category = "Self-Attention Blocks"
            elif "mlp" in name or "feed_forward" in name or "h." in name and "mlp" not in name:
                category = "Feed-Forward Networks (MLP)"
            elif "norm" in name or "ln_" in name:
                category = "Layer Normalizations"
            else:
                category = "Other / Miscellaneous"
            
            breakdown[category] = breakdown.get(category, 0) + num

        print("\n" + "=" * 78)
        print(" 🧠 MODEL ARCHITECTURE & PARAMETER BREAKDOWN")
        print("=" * 78)
        print(f" • Model Type           : {self.m_cfg.model_type.upper()}")
        print(f" • Hidden Size (h)      : {self.m_cfg.hidden_size}")
        print(f" • Hidden Layers (L)    : {self.m_cfg.layers}")
        print(f" • Attention Heads      : {self.m_cfg.heads} (KV Heads: {self.m_cfg.kv_heads})")
        print(f" • Intermediate Size    : {self.m_cfg.intermediate}")
        print(f" • Max Context Window   : {self.m_cfg.seq_len} tokens")
        print("-" * 78)
        print(f" | Component Breakdown                      | Parameter Count | % of Total |")
        print(f" | :---                                     | ---:            | ---:       |")
        
        for cat, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_params) * 100 if total_params > 0 else 0
            print(f" | {cat:<40} | {count:>13,} | {pct:>9.2f}% |")
        
        print("-" * 78)
        print(f" • **Total Parameters**     : {total_params:,}")
        print(f" • **Trainable Parameters** : {trainable_params:,}")
        print("=" * 78 + "\n")

    @staticmethod
    def _print_oom_guidance(batch_size, seq_len):
        print("\n" + "═" * 78)
        print(" 🚨 CUDA OUT OF MEMORY (OOM) ERROR DETECTED")
        print("═" * 78)
        print(f"   1. Lower your batch size: currently {batch_size} -> try reducing.")
        print(f"   2. Shorten your sequence length: currently {seq_len} -> try reducing.")
        print("   3. Disable model compilation by setting compile=False in your config.")
        print("═" * 78 + "\n")

    def _save_checkpoint(self, model, tokenizer, phase_name: str, global_step: int):
        """Saves a model checkpoint snapshot into a dedicated checkpoints subfolder."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"{phase_name}-step-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        unwrapped.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f" 💾 [Checkpoint Saved]: {checkpoint_dir}")

    def _generate_qualitative_samples(self, model, tokenizer, phase_name: str, active_sources: list):
        """Generates qualitative text continuations from canned seeds and live dataset samples."""
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        unwrapped.eval()

        print("\n" + "─" * 78)
        print(f" ✍️ QUALITATIVE GENERATION SAMPLES [{phase_name.upper()}]")
        print("─" * 78)

        # 1. Canned Seed Generations
        print(" [Canned Seeds Continuation]")
        for seed in self.canned_seeds:
            inputs = tokenizer(seed, return_tensors="pt").to(self.device)
            try:
                with torch.no_grad():
                    gen_tokens = unwrapped.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                clean_single_line = decoded.replace("\n", "\\n")
                if len(clean_single_line) > 120:
                    clean_single_line = clean_single_line[:117] + "..."
                print(f"   • Prompt: '{seed}'\n     Gen:    '{clean_single_line}'")
            except Exception as e:
                print(f"   • Prompt: '{seed}' -> Error generating: {e}")

        # 2. Live Dataset Row Continuations (First 100 tokens prompt -> completion)
        print("\n [Live Dataset Continuation Check]")
        for src in active_sources:
            try:
                raw_ds = load_dataset(src.name, src.subset, split="train", streaming=True)
                sample_item = next(iter(raw_ds))
                
                if src.preprocess_fn:
                    sample_item = src.preprocess_fn(sample_item)

                if src.is_sft and src.chat_template_fn:
                    prompt_text, answer_text = src.chat_template_fn(sample_item)
                    full_text = f"User: {prompt_text}\nAssistant: {answer_text}"
                else:
                    full_text = sample_item.get(src.text_column, sample_item.get("text", sample_item.get("content", "")))

                if not full_text:
                    continue

                tokenized_ids = tokenizer.encode(full_text, add_special_tokens=False)
                prompt_ids = tokenized_ids[:100]
                prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=True)

                inputs = tokenizer(prompt_str, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    gen_tokens = unwrapped.generate(
                        **inputs,
                        max_new_tokens=48,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                decoded_full = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                clean_continuation = decoded_full.replace("\n", "\\n")
                if len(clean_continuation) > 130:
                    clean_continuation = clean_continuation[:127] + "..."

                ds_short = src.name.split("/")[-1]
                print(f"   • Dataset [{ds_short}]:\n     Cont:   '{clean_continuation}'")
            except Exception as e:
                print(f"   • Dataset [{src.name}] -> Skip sample check: {e}")

        print("─" * 78 + "\n")
        unwrapped.train()

    def run_evaluation_suite(self, model, tokenizer, phase_name: str = "eval", global_step: int = 0, active_sources: list = None):
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        unwrapped.eval()
        
        print("\n" + "=" * 78)
        print(f" 📊 EVALUATION SUITE [{phase_name.upper()}] (Global Step: {global_step})")
        print("=" * 78)

        eval_configs = []
        for ds_spec in self.config.eval_datasets:
            source_obj = ds_spec.source
            if isinstance(source_obj, str):
                eval_configs.append(DatasetSourceConfig(name=source_obj, split="test", is_sft=False))
            else:
                eval_configs.append(DatasetSourceConfig(
                    name=source_obj.name,
                    subset=source_obj.subset,
                    split="test",
                    is_sft=source_obj.is_sft,
                    chat_template_fn=source_obj.chat_template_fn,
                    preprocess_fn=source_obj.preprocess_fn
                ))

        if not eval_configs:
            print(" [Eval Info] No evaluation datasets specified. Skipping evaluation suite.")
            print("=" * 78 + "\n")
            unwrapped.train()
            return 0.0, 0.0, 0.0

        eval_streams = DatasetRegistry.prepare_dataset_splits(
            eval_configs, tokenizer, self.m_cfg.seq_len, is_eval=True, holdout_ratio=self.config.holdout_ratio
        )

        dataset_metrics = []
        all_losses = []
        all_ems = []

        with torch.no_grad():
            for cfg, stream in zip(eval_configs, eval_streams):
                ds_name_short = cfg.name.split("/")[-1]
                ds_loss_sum = 0.0
                ds_batches = 0
                ds_ems = 0
                ds_samples = 0

                for batch_idx, sample in enumerate(stream):
                    if batch_idx > 30:
                        break
                    
                    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(self.device)
                    labels = torch.tensor([sample["labels"]], dtype=torch.long).to(self.device)

                    outputs = unwrapped(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    if not torch.isnan(loss):
                        ds_loss_sum += loss.item()
                        ds_batches += 1

                    if sample.get("is_sft", False):
                        try:
                            gen_out = unwrapped.generate(
                                input_ids=input_ids[:, :32],
                                max_new_tokens=24,
                                pad_token_id=tokenizer.eos_token_id
                            )
                            decoded_pred = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                            decoded_target = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                            
                            if decoded_target.strip() in decoded_pred.strip():
                                ds_ems += 1
                        except Exception:
                            pass
                        ds_samples += 1

                ds_avg_loss = ds_loss_sum / max(1, ds_batches)
                ds_perplexity = math.exp(min(ds_avg_loss, 20.0))
                ds_em_score = (ds_ems / max(1, ds_samples)) * 100.0 if ds_samples > 0 else 0.0

                dataset_metrics.append({
                    "dataset": ds_name_short,
                    "loss": ds_avg_loss,
                    "perplexity": ds_perplexity,
                    "em": ds_em_score,
                    "has_em": ds_samples > 0
                })

                all_losses.append(ds_avg_loss)
                if ds_samples > 0:
                    all_ems.append(ds_em_score)

                self.writer.add_scalar(f"eval_{phase_name}/{ds_name_short}/loss", ds_avg_loss, global_step)
                self.writer.add_scalar(f"eval_{phase_name}/{ds_name_short}/perplexity", ds_perplexity, global_step)
                if ds_samples > 0:
                    self.writer.add_scalar(f"eval_{phase_name}/{ds_name_short}/exact_match", ds_em_score, global_step)

        overall_loss = sum(all_losses) / max(1, len(all_losses))
        overall_perplexity = math.exp(min(overall_loss, 20.0))
        overall_em = sum(all_ems) / max(1, len(all_ems)) if all_ems else 0.0

        print("\n| Dataset | Loss | Perplexity | EM Score |")
        print("| :--- | :---: | :---: | :---: |")
        for m in dataset_metrics:
            em_str = f"{m['em']:.2f}%" if m["has_em"] else "N/A"
            print(f"| `{m['dataset']}` | {m['loss']:.4f} | {m['perplexity']:.2f} | {em_str} |")
        print(f"| **[AGGREGATE]** | **{overall_loss:.4f}** | **{overall_perplexity:.2f}** | **{overall_em:.2f}%** |")
        print("=" * 78 + "\n")

        self.writer.add_scalar(f"eval_{phase_name}/loss", overall_loss, global_step)
        self.writer.add_scalar(f"eval_{phase_name}/perplexity", overall_perplexity, global_step)
        self.writer.add_scalar(f"eval_{phase_name}/exact_match", overall_em, global_step)

        # Trigger qualitative sample generation and checkpoint saving
        if active_sources:
            self._generate_qualitative_samples(model, tokenizer, phase_name, active_sources)
        
        self._save_checkpoint(model, tokenizer, phase_name, global_step)

        unwrapped.train()
        return overall_loss, overall_perplexity, overall_em

    def _train_phase(self, phase_name: str, dataset_specs, model, tokenizer, optimizer, starting_step: int):
        if not dataset_specs:
            return starting_step

        phase_sources = []
        for ds_spec in dataset_specs:
            source_obj = ds_spec.source
            if isinstance(source_obj, str):
                base_cfg = DatasetSourceConfig(name=source_obj)
            else:
                base_cfg = DatasetSourceConfig(
                    name=source_obj.name,
                    subset=source_obj.subset,
                    split=source_obj.split,
                    is_sft=source_obj.is_sft,
                    text_column=source_obj.text_column,
                    preprocess_fn=source_obj.preprocess_fn,
                    chat_template_fn=source_obj.chat_template_fn
                )
            base_cfg.target_tokens = ds_spec.target_tokens
            base_cfg.weight = ds_spec.weight
            phase_sources.append(base_cfg)

        total_target_tokens = sum(s.target_tokens for s in phase_sources)
        streams = DatasetRegistry.prepare_dataset_splits(
            phase_sources, tokenizer, self.m_cfg.seq_len, is_eval=False, holdout_ratio=self.config.holdout_ratio
        )
        stream_weights = [s.weight for s in phase_sources]

        model.train()
        tokens_processed = 0
        step = 0
        global_step = starting_step
        start_time = time.time()
        next_eval_percent = self.config.eval_every_percent
        iters = [iter(s) for s in streams]

        pbar = tqdm(
            total=total_target_tokens, 
            desc=f"[{phase_name.upper()}]", 
            unit="tok", 
            unit_scale=True, 
            dynamic_ncols=True
        )

        try:
            while tokens_processed < total_target_tokens:
                chosen_stream_idx = random.choices(range(len(iters)), weights=stream_weights, k=1)[0]
                
                try:
                    sample = next(iters[chosen_stream_idx])
                except StopIteration:
                    streams[chosen_stream_idx] = iter(DatasetRegistry.prepare_dataset_splits(
                        [phase_sources[chosen_stream_idx]], tokenizer, self.m_cfg.seq_len, is_eval=False, holdout_ratio=self.config.holdout_ratio
                    )[0])
                    iters[chosen_stream_idx] = iter(streams[chosen_stream_idx])
                    sample = next(iters[chosen_stream_idx])

                input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)
                labels = torch.tensor(sample["labels"], dtype=torch.long).unsqueeze(0).to(self.device)

                try:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / self.config.grad_accum
                    loss.backward()
                except RuntimeError as e:
                    pbar.close()
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        self._print_oom_guidance(self.config.batch_size, self.m_cfg.seq_len)
                        sys.exit(1)
                    else:
                        raise e

                batch_tokens = input_ids.numel()
                tokens_processed += batch_tokens
                pbar.update(batch_tokens)

                if (step + 1) % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    elapsed = time.time() - start_time
                    tok_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    progress_pct = (tokens_processed / total_target_tokens) * 100
                    current_loss = outputs.loss.item()

                    self.writer.add_scalar(f"train_{phase_name}/loss", current_loss, global_step)
                    self.writer.add_scalar(f"train_{phase_name}/tokens_per_sec", tok_sec, global_step)

                    pbar.set_postfix(loss=f"{current_loss:.4f}", speed=f"{int(tok_sec)}tok/s")

                    if progress_pct >= next_eval_percent and progress_pct < 100:
                        pbar.write(f"\n[Triggering Milestone Evaluation, Generation & Checkpoint at {progress_pct:.1f}%]")
                        self.run_evaluation_suite(model, tokenizer, phase_name=f"{phase_name}_mid_{int(progress_pct)}pct", global_step=global_step, active_sources=phase_sources)
                        next_eval_percent += self.config.eval_every_percent

                step += 1

        finally:
            pbar.close()

        print(f"\n✔ Completed {phase_name.upper()} Phase! Running end-of-phase evaluation & checkpointing...")
        self.run_evaluation_suite(model, tokenizer, phase_name=f"{phase_name}_end", global_step=global_step, active_sources=phase_sources)
        
        return global_step

    def train(self, fresh: bool = True):
        print(f"Loading tokenizer: {self.config.tokenizer}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = self._build_model(vocab_size=len(tokenizer)).to(self.device)

        self._print_model_summary(model)

        if self.config.compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile() for high-speed training...")
            model = torch.compile(model)

        optimizer = AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        print(f"TEAPOT | SEQUENTIAL PIPELINE INITIALIZED")
        print(f"TensorBoard Logs: {self.writer.log_dir}")
        print(f"💡 [TENSORBOARD]: Run: tensorboard --logdir runs/\n")

        global_step = 0

        if self.config.pretrain_datasets:
            global_step = self._train_phase(
                phase_name="pretrain",
                dataset_specs=self.config.pretrain_datasets,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                starting_step=global_step
            )

        if self.config.sft_datasets:
            global_step = self._train_phase(
                phase_name="sft",
                dataset_specs=self.config.sft_datasets,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                starting_step=global_step
            )

        print("\nAll training phases complete! Running final comprehensive evaluation & checkpointing...")
        self.run_evaluation_suite(model, tokenizer, phase_name="final", global_step=global_step, active_sources=self.config.sft_datasets or self.config.pretrain_datasets)

        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        unwrapped.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"Model successfully saved to final output directory: {self.output_dir}!")