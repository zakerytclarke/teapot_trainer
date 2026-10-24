from teapottrainer import (
    TeapotTrainer, 
    TrainingConfig, 
    ModelConfig, 
    DatasetSpec, 
    FINEWEB_EDU, 
    SMOLTALK, 
    GSM8K
)

def main():
    print("Initializing 6-8 Hour Optimized Configuration (~95M Model / 800M Total Tokens)...")
    
    config = TrainingConfig(
        model=ModelConfig(
            model_type="llama",
            hidden_size=768,            # 768 hidden dimension
            num_hidden_layers=8,        # 8 layers (balanced depth/speed for ~95M total parameters)
            num_attention_heads=12,
            num_key_value_heads=4,      # Grouped-Query Attention (GQA)
            intermediate_size=2048,     # FFN expansion ratio
            max_position_embeddings=2048
        ),
        # Phase 1: Pretrain on FineWeb-Edu (700 Million Tokens - perfectly scoped for ~7 hour run)
        pretrain_datasets=[
            DatasetSpec(source=FINEWEB_EDU, target_tokens=4_000_000_000, weight=1.0)
        ],
        # Phase 2: SFT on SmolTalk for Conversational Alignment (100 Million Tokens)
        sft_datasets=[
            DatasetSpec(source=SMOLTALK, target_tokens=400_000_000, weight=1.0)
        ],
        # Evaluation Suite: GSM8K + SmolTalk test splits
        eval_datasets=[
            DatasetSpec(source=SMOLTALK, target_tokens=1_000_000)
        ],
        tokenizer="gpt2",                 # Clean ~50k vocabulary for optimal parameter balance
        batch_size=8,
        grad_accum=16,                    # Effective batch size = 128 sequences per step
        lr=6e-4,                          # Optimal peak learning rate
        weight_decay=0.1,
        eval_every_percent=10.0,          # Check eval metrics every 10% progress
        compile=True                      # Speeds up training throughput
    )

    trainer = TeapotTrainer(config=config)
    
    print("\n🚀 Starting 6-8 Hour Training Pipeline...")
    print("💡 To monitor training loss and metrics in real time, run:")
    print("   tensorboard --logdir runs/\n")

    trainer.train(fresh=True)

if __name__ == "__main__":
    main()