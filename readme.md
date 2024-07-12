# DeLA v2: Specialized Point Set Embedding

DeLA v2 is faster, more memory efficient and stronger in expressivity.
(Don't expect faster training though, there are many reasons: CPU bottleneck; Model scaled; CUDA kernels in fp32; prolonged steps...)

There will be no paper for DeLA v2. The method will be given concisely below. For technical details, please refer to the code.

And again, you're welcome to open an issue if you have any questions.

## Changes

I assume that you are familiar with [DeLA v1](https://github.com/Matrix-ASC/DeLA).
I'll omit things mentioned there.

For pytorch installation, I now recommend using their website's pip command. Code should run well on latest version even though I'm currently using 2.0.1.

### Model Architecture

[Specialized point set embedding (SPSE)](spse.md) is utilized for spatial encoding. This makes the network about 1.5x faster and saves 30% GPU memory.

[Specialized position aware max pooling (SPAMP)](spamp.md) is built upon SPSE and utilized for feature aggregation. This strengthens spatial awareness (relative coordinates prediction loss is well reduced). It has minor effect on training loss, but does help generalization under good regularization.

An MLP is used to transform SPSE generated spatial encoding to standard latent feature (as opposed to interpretable feature which seems to work badly with skip connection).

For classification, height is deleted for simplicity. At the last stage, features are mean + std pooled, and then fed to bn and a single full connected layer for logits.

Minor config settings.

### Training Methodology

Stronger data augmentations are used. Please check dataset scripts.

Switched to bf16 training. It eases method development and testing as it does not require special care for underflow.
Regarding performance, I guess bf16 and fp16 should be the same but I can't know for sure as I didn't run tests on fp16.
If your GPU needs fp16, you can modify code according to DeLA v1. I'm roughly sure no other special care is needed.

### Performance & Training time and memory requirement

Here I list mean $\pm$ std of at least 3 random runs.

Training time is measured on Ubuntu 22.04 with an RTX 4090 GPU and a 13600k CPU (power limited to 125w).

Memory consumption is checked from nvidia-smi after one epoch.

ModelNet40 and ShapeNetPart are removed due to performance saturation.

### S3DIS

mIoU:       74.44 $\pm$ 0.46

Time:       around 110 min, 140 min (with gradient checkpoint)

Memory:     6558 MB, 3418 MB (with gradient checkpoint)

pretrained (same model, test script has stochasticity):
test:  acc: 0.9262 || macc: 0.8121 || miou: 0.7551 || iou: [0.9545, 0.9861, 0.878, 0.0, 0.56, 0.6283, 0.8144, 0.8433, 0.9349, 0.8715, 0.8164, 0.8623, 0.6664]
test:  acc: 0.9258 || macc: 0.8115 || miou: 0.7543 || iou: [0.954, 0.986, 0.8777, 0.0, 0.5603, 0.6266, 0.8144, 0.8418, 0.9344, 0.8658, 0.8158, 0.8649, 0.6645]
test:  acc: 0.926 || macc: 0.8119 || miou: 0.7548 || iou: [0.9542, 0.9861, 0.8778, 0.0, 0.5599, 0.6268, 0.8161, 0.8429, 0.9341, 0.8681, 0.8165, 0.864, 0.6658]

### ScanNet v2

mIoU:       77.25 $\pm$ 0.28

Time:       around 450 min, 480 min (with gradient checkpoint)

Memory:     21232 MB, 10076 MB (with gradient checkpoint)

pretrained (same model, test script has stochasticity):
test:  acc: 0.9208 || macc: 0.8462 || miou: 0.7754 || iou: [0.8776, 0.9556, 0.7327, 0.8103, 0.9186, 0.7967, 0.8043, 0.7233, 0.7201, 0.8601, 0.3315, 0.7167, 0.746, 0.8043, 0.7515, 0.7429, 0.9534, 0.7054, 0.8974, 0.6607]
test:  acc: 0.9207 || macc: 0.8455 || miou: 0.7748 || iou: [0.8775, 0.9556, 0.733, 0.8106, 0.9188, 0.7975, 0.8046, 0.7216, 0.7198, 0.8612, 0.3316, 0.7169, 0.7434, 0.8021, 0.7519, 0.7361, 0.9531, 0.7025, 0.8969, 0.6603]

### ScanObjectNN

OA:         91.53 $\pm$ 0.40    (pretrained 92.05 bf16 checkpoint 92.02)

mAcc:       90.64 $\pm$ 0.44

Time:       around 35 min

Memory:     1678 MB

## Undone

SPSE and SPAMP CUDA kernels are in f32. Likely won't update it.

## P.S.

Owning only one GPU is a regularization that leads me to this solution.

Would I do better if I own less?

Anyway, appreciate you taking the time.
