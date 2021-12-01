from mmcv.runner import Hooks, Hook, master_only
import nsml

@HOOKS.register_module()
class MyHook(Hook):

    @master_only
    def after_train_epoch(self, runner):
        cur_epoch = runner.epoch + 1
        nsml.save(cur_epoch)