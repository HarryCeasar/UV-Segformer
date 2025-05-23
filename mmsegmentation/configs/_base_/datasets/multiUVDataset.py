# 数据处理 pipeline
# 同济子豪兄 2023-6-28

# 数据集路径
dataset_type = 'UrbanVillageDataset' # 数据集类名
data_root = 'MultiDataset/guangzhou' # 数据集路径（相对于mmsegmentation主目录）

# 输入模型的图像裁剪尺寸，一般是 128 的倍数，越小显存开销越少
crop_size = (512, 512)

# 训练预处理
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortionLimitedChannels'),
    dict(type='PackSegInputs')
]

# 测试预处理
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA后处理
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadSingleRSImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# 训练 Dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='multiband/train', seg_map_path='ann/train'),
        pipeline=train_pipeline))

# 验证 Dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='multiband/val', seg_map_path='ann/val'),
        pipeline=test_pipeline))

# 测试 Dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='multiband/test', seg_map_path='ann/test'),
        pipeline=test_pipeline))

# 验证 Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# 测试 Evaluator
test_evaluator = val_evaluator