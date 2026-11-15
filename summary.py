import torch

from ecg.models import ECGHybridClassifier


def print_model_summary(cfg: dict):
    dummy = torch.zeros(1, cfg["num_leads"], cfg["seq_len"])
    model = ECGHybridClassifier(cfg)
    model.eval()
    with torch.no_grad():
        cnn_out = model.cnn_encoder(dummy)
        tokens = model.tokenizer(cnn_out)
        tf_out = model.transformer(tokens)
        temporal_features = tf_out[:, 0, :]
        cnn_features = model.cnn_pool(cnn_out.mean(dim=1)).squeeze(-1)
        fused = torch.cat([temporal_features, cnn_features], dim=-1)
        logits = model.classifier(fused)

    print("\n  Model shapes (batch=1):")
    print(f"  Input                : {tuple(dummy.shape)}")
    print(f"  CNN encoder output   : {tuple(cnn_out.shape)}")
    print(f"  LeadTokenizer output : {tuple(tokens.shape)}")
    print(f"  Transformer output   : {tuple(tf_out.shape)}")
    print(f"  Fusion (CLS + CNN)   : {tuple(fused.shape)}")
    print(f"  Logits               : {tuple(logits.shape)}")
