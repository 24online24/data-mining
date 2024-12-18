def creare_set_validare():
    import os
    import pathlib
    import shutil
    import random
    base_dir = pathlib.Path("./aclImdb")
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(252).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname, val_dir / category / fname)


creare_set_validare()