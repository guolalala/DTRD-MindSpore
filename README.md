# Towards Long-delayed Sparsity: Learning a Better Transformer through Reward Redistribution

本仓库是论文 [Towards Long-delayed Sparsity: Learning a Better Transformer through Reward Redistribution](https://www.ijcai.org/proceedings/2023/0522.pdf) 的MindSpore代码实现。

## Requirements

- atari_py==0.2.9
- mindspore==2.0.0
- numpy==1.21.6
- opencv_contrib_python_headless==4.7.0.72
- opencv_python==4.8.1.78
- opencv_python_headless==4.7.0.72
- PyYAML==6.0.1


依赖库可以通过以下命令安装：
```bash
pip install -r requirements.txt
```

## Configs

配置文件位于 `configs` 文件夹下，主要包含以下内容：

- `dtrd_train.yaml`：训练配置文件，包含训练参数、优化器、学习率衰减策略等。
- `dtrd_test.yaml`：测试配置文件，包含测试参数、模型路径等。

## Training

训练脚本位于 `train.py` 文件中，可以通过以下命令进行训练：

```bash
python -u train.py --model DTRD --data Qbert -c ./configs/dtrd/dtrd_train.yaml --do_train
```

我们也提供了一系列的训练脚本，位于 `scripts` 文件夹下，可以通过以下命令进行训练：

```bash
bash scripts/DTRD_train.sh
```

## Evaluation

进行测试的时候首先需要运行以下命令来加载ROMS：

```bash
python -m atari_py.import_roms ./Roms
```

可以通过以下命令进行评估：

```bash
python -u train.py --model DTRD --data Qbert -c ./configs/dtrd/dtrd_test.yaml
```

我们也提供了一系列的评估脚本，位于 `scripts` 文件夹下，可以通过以下命令进行评估：

```bash
bash scripts/DTRD_test.sh
```

## Results

测试结果如下：

```
Device: GPU
Args in experiment:
Namespace(activation='gelu', attn='prob', batch_size=128, c_out=7, ceof=0.5, checkpoint_path='./checkpoints/test_ckpt/dtrd_best.ckpt', checkpoints='./checkpoints/', ckpt_path='./checkpoints/test_ckpt/dtrd_best.ckpt', cols=None, config='./configs/dtrd/dtrd_test.yaml', config_file='', context_length=30, d_ff=2048, d_layers=1, d_model=512, data='Qbert', data_dir='./data/game_data/', data_name='Qbert', data_path='ETTh1.csv', dec_in=7, des='test', detail_freq='h', device='GPU', device_num=None, devices='0,1,2,3', distil=True, distribute=False, do_predict=False, do_train=False, dropout=0.05, e_layers=2, embed='timeF', enc_in=7, epochs=5, factor=5, features='M', freq='h', game='Qbert', gpu=0, inverse=False, itr=1, label_len=48, learning_rate=0.0001, longforecast=0, loss='mse', lradj='type1', mix=True, model='DTRD', model_name='DTRD', model_type='reward_conditioned', n_heads=8, num_buffers=50, num_steps=500000, num_workers=0, output_attention=False, padding=0, patience=3, pred_len=24, pretrained=True, rank_id=None, root_path='./data/ETT/', s_layers='3,2,1', seed=42, seq_len=96, target='OT', train_epochs=6, trajectories_per_buffer=10, use_amp=False, use_gpu=True, use_multi_gpu=False)
02/21/2024 18:10:22 - INFO - models.DTRD -   number of parameters: 1.872128e+06
Eval score: 3287
TEST TIME: 585.6708333492279
```