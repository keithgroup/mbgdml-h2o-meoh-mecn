# Training logs

Each directory contains training logs for a specific system and machine learning method.
`idxs` directories contain npy files of training, validation, and test indices with respect to the dataset the model is being trained on.
This is useful for training different ML models on the same structures.

All final models are copied into the root `models` directory.
GAP models are removed from this this directory due to their large file size.
