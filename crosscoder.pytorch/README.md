- [x] Rerun dead neurons ref to check whether the percentage of dead neurons
  stay the same
    - @RESULT: Confirm that the percentage of dead neurons stay the same (2284
      neurons when decoder weights is inited at 0.08)
    - @OBS: the reference implementation has higher number of dead neurons and
      not as sparse as `CrossCoderV1ENormalizeKaimingInitTranspose0.08-2 `
- [x] Check the effect of decoder init norm on the number of dead neurons for
  the reference model
    - @RESULT:
        - 0.03: 4669
        - 0.1: 2088
        - 0.12: 1944
        - 0.14: 2164
        - 0.16: 2480
        - 0.18: 3398
        - 1: 1.18e+4
        - There exists a sweet spot for the decoder init norm, such that the
          number of dead neurons can decrease.
        - The decoder weight also plays an important role in calculating the
          regularization term.
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
- [x] Check if loading dataset is efficient
    - It's currently not. It has error in that even doesn't iterate through the
      whole dataset.
    - You want to achieve:
        - Multiple processeses sharing the same block of data on RAM.
        - Preload the data from disk to RAM during computational processes, so
          that we decrease I/O bottleneck from computation time.
        - Preload the data from RAM to GPU memory during computational
          processes, so that we decrease I/O bottleneck from computation time.
        - Add more SSD to store read-intensive data.
        - Add more RAM to reduce the frequency of copying data.
    - @RESULT: fix a bug in data loading that prevents the whole dataset from
      being completely iterated.
- [x] Incorprate the explained variance into metrics
- [x] Rerun trainin V2. It has training metric tracking and correct validation
  dataset
- [ ] The reconstruction is shit. It's even worse than just switching layer.
    - On median, an input has value of 1.2796. On median, we mis-predict by
      1.2575. So, we are off by nearly 100%.
    - Need to improve the reconstruction so that it can effectively recovers
      the original input.
    - [ ] Don't involve the decoder in the reg section
- [ ] @TRAINING Track the weights evolution.
    - Encoder
    - Decoder
    - @TRAINING: currently in training. Observe how the weight norm evolves.
    - @TRAINING: also, the training l0 metrics also have a window of 12000
      steps rather than full training epoch to see the evolution overtime and
      don't have any number dominiates from the beginning to the end of the
      training epoch.
- [x] @ABANDON Implement cfg file. Maybe reuse from Hydra.
    - Not much benefits compared to declaring in `__init__` 
    - You can allow flexible params in `__init__` by using `*args` and `**kwargs`
- [x] Overfit small training set
    - -> It's even hard to overfit on small amount of data. It takes so much
      time, around 400 iterations, to overfit a single batch of data.
- [ ] Remove the first token.
    - The first token in the context usually has abnormally high max value.
      Maybe this high value makes training unstable, as the crosscoder will be
      incentivized to predict the first token correctly. Plus, because the
      first token has high value, its varations is also very high, making
      learning hard.
- [x] Track the gradient norms.
    - To answer the question: Why does the reconstruction loss not decrease much
    - The gradients are large and vary widely.
- Ideas to play around:
    - [ ] Add lr scheduler
    - [ ] Add lambda coefficient scheduler
    - [ ] Clip grad norm
- [ ] Run non-detach with different dec_init_norm value
- [ ] Use the reconstructed activations during text generation.
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
    - You can list remaining jobs, modifying the remaining jobs, or deleting
      the jobs.
    - You can list hardware accelerators on multiple machines.
    - You can push job so that it will be executed in a different machines.
    - Sounds like a more flexible vast.ai or prime intellect, or petals.
- [x] Prepare a full dataset.
- [x] Count the total number of tokens
    - Around 9.7 - 9.8 billion tokens  per file
    - So around 294 billion tokens in total for 30 files
- [ ] Run from the full dataset. -> Can implement and try out now. The full
  dataset is huge, so a partial dataset can be considered a lot. In fact, you
  can try out now at the moment, and then gradually add more data during
  training.
    - [x] How much text lines can an LLM process
       - 6 lines a batch, at 1024 context length
    - [ ] Set maximum number of tokens: 1 billion token
    - Train inside a "while" loop.
    - Data is stored inside a buffer (ideally VRAM), and then consumed
      immediately, kind of JIT
    - Select batch of 6, with 1024 context lenght.
    - Minimize costly data transfer.
    - Free up VRAM.
- [x] Load the binary dataset randomly.
- [ ] Preliminary implement the steering capability to try this out.
- [x] Change the separator to 65535, because the endoftext token exists in the
  dataset.
    - Not important.
      - 00.jsonl at line 2008

==== Number of data:

- 00.jsonl: 5899215
- 07.jsonl: 5901316
- 28.jsonl: 5900136
- 16.jsonl: 5900781

====

It might be naive to expect the reconstructed activations can be close to the
original input, when the intermediate features are sparse. Imagine, if you
compress the original activations, from 768 to 40, and then you have to
accurately reconstruct into 768 activations -> Sounds like possible. Another
question is which 40 features that the model choose to optimize.

<?Q> Along the way, does the training increase the absolute value of any
feature, or it just gradually and consistently scalp down the features whether
or not it hurts model understanding?

<?Q> What if we just ignore negative values, rather than clip them to zero with
Relu. Maybe doing so can help with gradient?
