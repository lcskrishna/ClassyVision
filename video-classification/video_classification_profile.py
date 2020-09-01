import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import torch
from classy_vision.dataset import build_dataset

video_dir = "/data/datasets/ucf101/UCF-101"
splits_dir = "/data/datasets/ucf101/ucfTrainTestlist/"
metadata_file = "ucf101_metadata.pt"

def get_percentage(time_us,total_time_us):
    percentage = (time_us * 100.0)/total_time_us
    return percentage

#Make sure the profile results are sorted by cuda_time.
def extract_profile_information_to_csv(profile_results, sort_by=None):
    from torch.autograd.profiler import EventList
    fs = open("autograd_profile.csv", "w")
    fs.write('sep=|')
    fs.write('\n')
    if sort_by is not None:
        profile_results = EventList(sorted(profile_results, key=lambda evt : getattr(evt, sort_by), reverse=True), use_cuda=True)

    self_cpu_time_total = sum([event.self_cpu_time_total for event in profile_results])
    cuda_time_total = sum([evt.cuda_time_total for evt in profile_results])
    header_str = "KernelName" + "|" + "GPU Percentage(%)" + "|" + "GPU Total Time" + "|" + "GPU Average" + "|" + "NumOfCalls"
    fs.write(header_str)
    fs.write("\n")
    for j in range(len(profile_results)):
        event = profile_results[j]
        output_str = ""
        output_str = output_str + event.key + "|" ## Name.
        percentage_gpu = str(get_percentage(event.cuda_time_total, cuda_time_total))
        output_str = output_str + percentage_gpu + "|"
        output_str = output_str + str(event.cuda_time_total) + "|"
        output_str = output_str + event.cuda_time_str + "|"
        output_str = output_str + str(event.count)
        fs.write(output_str)
        fs.write("\n")
    fs.close()

datasets = {}
## Train dataset.
datasets['train'] = build_dataset({
    "name" : "ucf101",
    "split": "train",
    "num_workers" : 0,
    "batchsize_per_replica" : 8,
    "use_shuffle" : True,
    "num_samples" : 64,
    "clips_per_video" : 1,
    "frames_per_clip" : 8,
    "video_dir" : video_dir,
    "splits_dir" : splits_dir,
    "metadata_file" : metadata_file,
    "fold" : 1,
    "tranforms" : {
        "video" : [
            {
               "name" : "video_default_augment",
               "crop_size" : 112,
               "size_range" : [128, 160],
            }
        ]
    }
})


datasets['test'] = build_dataset({
    "name" : "ucf101",
    "split": "test",
    "num_workers" : 0,
    "batchsize_per_replica" : 10,
    "use_shuffle" : False,
    "num_samples" : 80,
    "clips_per_video" : 10,
    "frames_per_clip" : 8,
    "video_dir" : video_dir,
    "splits_dir" : splits_dir,
    "metadata_file" : metadata_file,
    "fold" : 1,
    "transforms" : {
        "video" : [
            {
                "name" : "video_default_no_augment",
                "size" : 128
            }
        ]
    }
})

print ("INFO: Dataset loaded.")

## Define Resnext3D model.
from classy_vision.models import build_model
model = build_model({
    "name" : "resnext3d",
    "frames_per_clip": 8,
    "input_planes" : 3,
    "clip_crop_size" :  112,
    "skip_transformation_type" : "postactivated_shortcut",
    "residual_transformation_type" : "basic_transformation",
    "num_blocks" : [2, 2, 2, 2],
    "input_key" : "video",
    "stage_planes" : 64,
    "num_classes" : 101
})

from classy_vision.heads import build_head
from collections import defaultdict

unique_id = "default_head"
head = build_head({
    "name": "fully_convolutional_linear",
    "unique_id": unique_id,
    "pool_size": [1, 7, 7],
    "num_classes": 101,
    "in_plane": 512    
})
fork_block = "pathway0-stage4-block1"
heads = {fork_block: [head]}
model.set_heads(heads) 

print ("INFO: Model built.")

## Meters.
from classy_vision.meters import build_meters, AccuracyMeter, VideoAccuracyMeter

meters = build_meters({
    "accuracy" : {
        "topk" : [1,5]
    },
    "video_accuracy" : {
        "topk" : [1,5],
        "clips_per_video_train" : 1,
        "clips_per_video_test" : 10,
    }
})

## Building the task.
from classy_vision.tasks import ClassificationTask
from classy_vision.optim import build_optimizer
from classy_vision.losses import build_loss

loss = build_loss({"name" : "CrossEntropyLoss"})
optimizer = build_optimizer({
        "name" : "sgd",
        "param_schedulers" : {
            "lr" : {
                    "name" : "multistep",
                    "values" : [0.005, 0.0005],
                    "milestones" : [1]
                   }
        },
        "num_epochs" : 10,
        "weight_decay" : 0.0001,
        "momentum" : 0.9,
})

num_epochs = 10
task = (
    ClassificationTask()
    .set_num_epochs(num_epochs)
    .set_loss(loss)
    .set_model(model)
    .set_optimizer(optimizer)
    .set_meters(meters)
)

for phase in ["train" , "test"]:
    task.set_dataset(datasets[phase], phase)

#task.set_dataloader_mp_context("fork")

### Start training.
print ("INFO: Start training.")
import time
import os

from classy_vision.trainer import LocalTrainer
from classy_vision.hooks import CheckpointHook
from classy_vision.hooks import LossLrMeterLoggingHook

hooks = [LossLrMeterLoggingHook(log_freq=1)]
checkpoint_dir = f"./output_checkpoint_{time.time()}"
os.mkdir(checkpoint_dir)
hooks.append(CheckpointHook(checkpoint_dir, input_args={}))

task = task.set_hooks(hooks)

trainer = LocalTrainer()
with torch.autograd.profiler.profile(use_cuda=True, enabled=True) as prof:
    trainer.train(task)
extract_profile_information_to_csv(prof.key_averages(group_by_input_shape=True), sort_by="cuda_time_total")
