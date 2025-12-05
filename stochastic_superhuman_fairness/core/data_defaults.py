DEFAULT_DATA_CONFIGS = {
    "adult": {
        "label_col": "income",
        "protected_attrs": ["sex", "race"],
        "normalize": True,
        "one_hot": True,
        "train_ratio": 0.8,
        "cache_dir": "./data",
    },
    "compas": {
        "label_col": "two_year_recid",
        "protected_attrs": ["sex", "race"],
        "normalize": True,
        "one_hot": True,
        "train_ratio": 0.8,
        "cache_dir": "./data",
    },
}
