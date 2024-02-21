"""train scripts"""
import os
import mindspore as ms
from config import parse_args
import argparse
import yaml
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank, init

from models.Exp_DTRD import Exp_DTRD
from utils.random import set_seed

model_entrypoints = {
    'DTRD': Exp_DTRD,
}

class MindSeqModel():
    def __init__(self,exp,args):
        self.exp = exp
        # self.model = None
        self.args = args
        self.setting = "Setting not set"

    def train(self,itr=0):
        self.exp.train()
    
    def test(self,itr=0):
        self.exp.test()
    


def is_model(model_name):
    return model_name in model_entrypoints.keys()

def create_model(
        model_name: str,
        data_name: str,
        pretrained: bool = False,
        checkpoint_path: str='',
        config_file: str='',
        **kwargs,
):
    if checkpoint_path == "" and pretrained:
        raise ValueError("checkpoint_path is mutually exclusive with pretrained")
    
    # 创建一个命名空间对象
    args = argparse.Namespace(
        model_name=model_name,
        data_name=data_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        config_file = config_file,
        **kwargs  # 添加 **kwargs 中的参数
    )
    if args.config_file:
        with open(config_file,'r') as f:
            config_data = yaml.safe_load(f)
            args.__dict__.update(config_data)
    if hasattr(args, 'model'):
        args.model = model_name
    if hasattr(args, 'data'):
        args.data = data_name
    data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'weather':{'data':'weather.csv','T':'OT','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1]},
    }
    if hasattr(args, 'data'):
        if args.data in data_parser.keys():
            data_info = None
            if hasattr(args, 'data'):
                data_info = data_parser[args.data]
            if hasattr(args, 'data_path') and hasattr(args, 'target'):
                args.data_path = data_info['data']
                args.target = data_info['T']
            if hasattr(args, 'enc_in') and hasattr(args, 'dec_in') and hasattr(args, 'c_out') and hasattr(args, 'features'):
                args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    print('Args in experiment:')
    print(args)

    # set mode
    if hasattr(args, 'device'):
        ms.set_context(device_target=args.device)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    if hasattr(args, 'distribute'):
        if args.distribute:
            init()
            if hasattr(args, 'device_num') and hasattr(args, 'rank_id'):
                args.device_num = get_group_size()
                args.rank_id = get_rank()
            if hasattr(args, 'device_num'):
                ms.set_auto_parallel_context(
                    device_num=args.device_num,
                    parallel_mode="data_parallel",
                    gradients_mean=True,
                    # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
                )
        elif hasattr(args, 'device_num') and hasattr(args, 'rank_id'):
            args.device_num = None
            args.rank_id = None
    if hasattr(args, "seed"):
        set_seed(args.seed)

    # Check model_name
    if not is_model(args.model_name):
        raise RuntimeError(f'Unknow model {args.model_name}, options:{model_entrypoints.keys()}')
    
    exp = model_entrypoints[args.model_name](args)
    Exp = MindSeqModel(exp,args)
    
    return Exp



def train(args):
    """main train function"""
    # set mode
    ms.set_context(device_target=args.device)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    print("Device:", args.device)

    # create Exp
    exp = create_model(model_name=args.model, data_name=args.data, \
                       pretrained=not args.do_train, checkpoint_path=args.ckpt_path,**vars(args))
    if args.do_train or args.ckpt_path=='':
        if not os.path.exists("./checkpoints/train_ckpt"):
            os.mkdir("./checkpoints/train_ckpt")
        for i in range(args.itr):
            exp.train(i)
    else:
        exp.test(0)

if __name__ == "__main__":
    my_args = parse_args()
    # 命令行和配置文件可同时输入，以命令行为准

    # core train
    train(my_args)
