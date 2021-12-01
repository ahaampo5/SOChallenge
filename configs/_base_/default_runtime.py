checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
<<<<<<< HEAD
custom_hooks = [dict(type='NumClassCheckHook')]
=======
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='MyHook')]
>>>>>>> 8e8b0258373baf9d9a76420023d861cd450cd7e4

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
