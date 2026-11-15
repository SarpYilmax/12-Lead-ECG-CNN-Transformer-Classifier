SEED = 26

PRIMARY_DATA_BASE = "/kaggle/input/datasets/sarpylmz/teknofest-second-stage/teknofest_ECG_2026_dataset"
PHYSIONET_BASE = "/kaggle/input/datasets/sarpylmz/sarp-physionet-ecg/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"

CFG = {
    "data_root": PRIMARY_DATA_BASE,
    "physionet_root": PHYSIONET_BASE,
    "train_csv": f"{PRIMARY_DATA_BASE}/train.csv",
    "val_csv": f"{PRIMARY_DATA_BASE}/validation.csv",
    "test_csv": f"{PRIMARY_DATA_BASE}/test_public.csv",

    "num_classes": 5,
    "class_names": ["Normal", "AFIB", "AFL", "LBBB", "RBBB"],
    "num_leads": 12,
    "seq_len": 5000,

    "stem_out": 32,
    "cnn_channels": [64, 128, 256],
    "cnn_kernel": 7,
    "cnn_stride": 2,
    "cnn_dropout": 0.2,
    "se_reduction": 16,

    "d_model": 256,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 512,
    "tf_dropout": 0.1,
    "max_seq_enc": 4096,

    "epochs": 50,
    "batch_size": 32,
    "lr": 7e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "patience": 18,
    "checkpoint_path": "best_model.pt",

    "focal_gamma": 2.7,
    "label_smoothing": 0.05,
    "focal_alpha": [0.1, 0.35, 0.35, 0.1, 0.1],

    "finetune_epochs": 6,
    "finetune_lr": 5e-5,
    "finetune_checkpoint_path": "best_model_afib_afl_finetuned.pt",
}

PHYSIONET_CODE_MAP = {
    "426783006": 0,
    "427393009": 0,
    "164889003": 1,
    "164890007": 2,
    "164909002": 3,
    "164934002": 4,
}

PHYSIONET_TARGET_CLASSES = [1, 2]
