- [x] Rerun dead neurons ref to check whether the percentage of dead neurons stay the same
    - @RESULT: Confirm that the percentage of dead neurons stay the same (2284 neurons when decoder weights is inited at 0.08)
- [x] Check the effect of decoder init norm on the number of dead neurons for the reference model
    - @RESULT:
        - 0.03: 4669
        - 0.1: 2088
        - 0.12: 1944
        - 0.14: 2164
        - 0.16: 2480
        - 0.18: 3398
        - 1: 1.18e+4
        - There exists a sweet spot for the decoder init norm, such that the number of dead neurons can decrease.
        - The decoder weight also plays an important role in calculating the regularization term.
- [x] Check dead neurons of 1A
    - @Result -> Quite good.
        - No dead neurons
        - Why is that? This is so weird.
- [x] Use correct transpose version for kaiming init
    - @Result -> Dead neurons become 5128
- [x] Normalize the weight after kaiming init
    - @Result:
        - 0.12: Dead neurons become 65. It is an acceptable number. 0.0228
          percent of features are activated on average.
- [ ] The reconstruction is shit. It's even worse than just switching layer.
    - On median, an input has value of 1.2796. On median, we mis-predict by 1.2575. So, we are off by nearly 100%.
    - [ ] Don't involve the decoder in the reg section
- [ ] Incorprate the explained variance into metrics
- [x] Check if loading dataset is efficient
    - It's currently not. It has error in that even doesn't iterate through the whole dataset.
    - You want to achieve:
        - Multiple processeses sharing the same block of data on RAM.
        - Preload the data from disk to RAM during computational processes, so that we decrease I/O bottleneck from computation time.
        - Preload the data from RAM to GPU memory during computational processes, so that we decrease I/O bottleneck from computation time.
        - Add more SSD to store read-intensive data.
        - Add more RAM to reduce the frequency of copying data.
    - @RESULT: fix a bug in data loading that prevents the whole dataset from being completely iterated.
- [ ] Incorprate the explained variance into metrics

- [ ] Experiment management system, where you can submit your job queue
    - [ ] Can get some hands on experience with Slurm
    - Experiment management is quite tricky:
        - Correct data when the job is submitted.
        - Correct code when the job is submitted.
        - Correct configuration when the job is submitted.
        - Have the correct environment to run the training.
        - Multi-job management (when running jobs in parallel).
    - The job progress will be shown in the service (in the terminal) and in
      tensorboard...
    - You can have another comment to push the job to a queue.
    - And then the services will execute that job.
    - You can list remaining jobs, modifying the remaining jobs, or deleting the jobs.
    - You can list hardware accelerators on multiple machines.
    - You can push job so that it will be executed in a different machines.
    - Sounds like a more flexible vast.ai or prime intellect, or petals.


## Rerun with correct dataloader

```
===
CrossCoderV1ENormalizeKaimingInitTranspose0.08-2
Epoch 0: 100%|█████████████████████████████████████████████████| 3782/3782 [26:02<00:00,  2.42it/s, v_num=08-2{'total_inactive': 0, 'mean_active': 627.482666015625, 'mean_active_pct': 0.05106467008590698, 'max_active': 4927, 'max_active_pct': 0.4009602864583333, 'min_active': 0, 'min_active_pct': 0.0, 'total_dead_neurons': 0, 'total_neurons': 12288, 'pct_dead_neurons': 0.0, 'mean_explained_variance': 0.5863010192948266, 'mean_explained_variance_a': 0.5861430729915584, 'mean_explained_variance_b': 0.585623548583475, 'residual': 6.7730317}
Epoch 1: 100%|█████████████████████████████████████████████████| 3782/3782 [25:41<00:00,  2.45it/s, v_num=08-2{'total_inactive': 0, 'mean_active': 245.587890625, 'mean_active_pct': 0.01998599370320638, 'max_active': 3659, 'max_active_pct': 0.2977701822916667, 'min_active': 0, 'min_active_pct': 0.0, 'total_dead_neurons': 32, 'total_neurons': 12288, 'pct_dead_neurons': 0.0026041666666666665, 'mean_explained_variance': 0.6309540859530341, 'mean_explained_variance_a': 0.6303754338082633, 'mean_explained_variance_b': 0.6306368779146244, 'residual': 6.034538}
Epoch 2: 100%|█████████████████████████████████████████████████| 3782/3782 [25:46<00:00,  2.45it/s, v_num=08-2{'total_inactive': 0, 'mean_active': 132.14794921875, 'mean_active_pct': 0.010754227638244629, 'max_active': 1688, 'max_active_pct': 0.13736979166666666, 'min_active': 0, 'min_active_pct': 0.0, 'total_dead_neurons': 40, 'total_neurons': 12288, 'pct_dead_neurons': 0.0032552083333333335, 'mean_explained_variance': 0.6719320795924736, 'mean_explained_variance_a': 0.6723307539715931, 'mean_explained_variance_b': 0.6709001010078673, 'residual': 5.3785796}
===
Epoch 0: 100%|█████████████████████████████████████████████████| 3782/3782 [27:56<00:00,  2.26it/s, v_num=ons2{'total_inactive': 0, 'mean_active': 552.849853515625, 'mean_active_pct': 0.04499103625615438, 'max_active': 8888, 'max_active_pct': 0.7233072916666666, 'min_active': 0, 'min_active_pct': 0.0, 'total_dead_neurons': 0, 'total_neurons': 12288, 'pct_dead_neurons': 0.0, 'mean_explained_variance': 0.5789166817380751, 'mean_explained_variance_a': 0.579682060631277, 'mean_explained_variance_b': 0.5775573415824318, 'residual': 6.8809013}
```
