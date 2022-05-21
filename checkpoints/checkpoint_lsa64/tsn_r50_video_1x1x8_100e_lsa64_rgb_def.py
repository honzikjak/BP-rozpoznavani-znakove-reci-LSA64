model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=64,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 200
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'C:/Users/honzi/mmaction2/checkpoints/checkpoint_lsa64/latest.pth'
workflow = [('train', 1), ('val', 1)]
dataset_type = 'VideoDataset'
data_root = 'C:/Users/honzi/mmaction2/data/lsa64/'
data_root_val = 'C:/Users/honzi/mmaction2/data/lsa64/'
ann_file_train = 'C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_train_list.txt'
ann_file_val = 'C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_val_list.txt'
ann_file_test = 'C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_test_list.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
data = dict(
    videos_per_gpu=3,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_train_list.txt',
        data_prefix='C:/Users/honzi/mmaction2/data/lsa64/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames', clip_len=1, frame_interval=1,
                num_clips=8),
            dict(type='DecordDecode'),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='VideoDataset',
        ann_file='C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_val_list.txt',
        data_prefix='C:/Users/honzi/mmaction2/data/lsa64/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='VideoDataset',
        ann_file='C:/Users/honzi/mmaction2/data/lsa64/lsa64_ann_test_list.txt',
        data_prefix='C:/Users/honzi/mmaction2/data/lsa64/',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='TenCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
work_dir = 'C:/Users/honzi/mmaction2/checkpoints/checkpoint_lsa64'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
