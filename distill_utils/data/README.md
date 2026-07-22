# Dataset Directory

Raw datasets and extracted video frames are not included in this supplement because they exceed the 50 MB archive limit and are subject to their original licenses.

Place datasets here using the relative paths expected by `utils.py`.

SSv2:

```text
SSv2/
  annot_train.json
  annot_val.json
  frame/
    <video_id>/
      <frame files>
```

UCF101 / miniUCF101:

```text
UCF101/
  ucf101_splits1.csv
  jpegs_112/
    <video_folder>/
      img_00001.jpg
      ...
```

HMDB51:

```text
HMDB51/
  hmdb51_splits.csv
  jpegs_112/
    <video_folder>/
      <frame files>
```

The SSv2 loader requires each video directory to contain exactly the number of frames requested by `FRAMES`.
