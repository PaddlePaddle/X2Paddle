## 实时语音克隆 - 中文/普通话
![mockingbird](https://user-images.githubusercontent.com/12797292/131216767-6eb251d6-14fc-4951-8324-2722f0cd4c63.jpg)

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)

### [English](README.md)  | 中文

### [DEMO VIDEO](https://www.bilibili.com/video/BV1sA411P7wM/)

## 特性
🌍 **中文** 支持普通话并使用多种中文数据集进行测试：aidatatang_200zh, magicdata, aishell3， biaobei，MozillaCommonVoice 等

🤩 **PyTorch** 适用于 pytorch，已在 1.9.0 版本（最新于 2021 年 8 月）中测试，GPU Tesla T4 和 GTX 2060

🌍 **Windows + Linux** 可在 Windows 操作系统和 linux 操作系统中运行（苹果系统M1版也有社区成功运行案例）

🤩 **Easy & Awesome** 仅需下载或新训练合成器（synthesizer）就有良好效果，复用预训练的编码器/声码器，或实时的HiFi-GAN作为vocoder

🌍 **Webserver Ready** 可伺服你的训练结果，供远程调用

### 1. 安装要求
> 按照原始存储库测试您是否已准备好所有环境。
**Python 3.7 或更高版本** 需要运行工具箱。

* 安装 [PyTorch](https://pytorch.org/get-started/locally/)。
> 如果在用 pip 方式安装的时候出现 `ERROR: Could not find a version that satisfies the requirement torch==1.9.0+cu102 (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2)` 这个错误可能是 python 版本过低，3.9 可以安装成功
* 安装 [ffmpeg](https://ffmpeg.org/download.html#get-packages)。
* 运行`pip install -r requirements.txt` 来安装剩余的必要包。
* 安装 webrtcvad `pip install webrtcvad-wheels`。

### 2. 准备预训练模型
考虑训练您自己专属的模型或者下载社区他人训练好的模型:
#### 2.1 使用数据集自己训练合成器模型（与2.2二选一）
* 下载 数据集并解压：确保您可以访问 *train* 文件夹中的所有音频文件（如.wav）
* 进行音频和梅尔频谱图预处理：
`python pre.py <datasets_root>`
可以传入参数 --dataset `{dataset}` 支持 aidatatang_200zh, magicdata, aishell3
> 假如你下载的 `aidatatang_200zh`文件放在D盘，`train`文件路径为 `D:\data\aidatatang_200zh\corpus\train` , 你的`datasets_root`就是 `D:\data\`

* 训练合成器：
`python synthesizer_train.py mandarin <datasets_root>/SV2TTS/synthesizer`

* 当您在训练文件夹 *synthesizer/saved_models/* 中看到注意线显示和损失满足您的需要时，请转到`启动程序`一步。

#### 2.2使用社区预先训练好的合成器（与2.1二选一）
> 当实在没有设备或者不想慢慢调试，可以使用社区贡献的模型(欢迎持续分享):

| 作者 | 下载链接 | 效果预览 | 信息 |
| --- | ----------- | ----- | ----- |
|@FawenYo | https://drive.google.com/file/d/1H-YGOUHpmqKxJ9FRc6vAjPuqQki24UbC/view?usp=sharing [百度盘链接](https://pan.baidu.com/s/1vSYXO4wsLyjnF3Unl-Xoxg) 提取码：1024  | [input](https://github.com/babysor/MockingBird/wiki/audio/self_test.mp3) [output](https://github.com/babysor/MockingBird/wiki/audio/export.wav) | 200k steps 台湾口音
|@miven| https://pan.baidu.com/s/1PI-hM3sn5wbeChRryX-RCQ 提取码：2021 | https://www.bilibili.com/video/BV1uh411B7AD/ | 150k steps 旧版需根据[issue](https://github.com/babysor/MockingBird/issues/37)修复

#### 2.3训练声码器 (可选)
对效果影响不大，已经预置3款，如果希望自己训练可以参考以下命令。
* 预处理数据:
`python vocoder_preprocess.py <datasets_root> -m <synthesizer_model_path>`
> `<datasets_root>`替换为你的数据集目录，`<synthesizer_model_path>`替换为一个你最好的synthesizer模型目录，例如 *sythensizer\saved_mode\xxx*


* 训练wavernn声码器:
`python vocoder_train.py <trainid> <datasets_root>`
> `<trainid>`替换为你想要的标识，同一标识再次训练时会延续原模型

* 训练hifigan声码器:
`python vocoder_train.py <trainid> <datasets_root> hifigan`
> `<trainid>`替换为你想要的标识，同一标识再次训练时会延续原模型

### 3. 启动程序或工具箱
您可以尝试使用以下命令：

### 3.1 启动Web程序：
`python web.py`
运行成功后在浏览器打开地址, 默认为 `http://localhost:8080`
<img width="578" alt="bd64cd80385754afa599e3840504f45" src="https://user-images.githubusercontent.com/7423248/134275205-c95e6bd8-4f41-4eb5-9143-0390627baee1.png">
> 注：目前界面比较buggy, 
> * 第一次点击`录制`要等待几秒浏览器正常启动录音，否则会有重音
> * 录制结束不要再点`录制`而是`停止`
> * 仅支持手动新录音（16khz）, 不支持超过4MB的录音，最佳长度在5~15秒
> * 默认使用第一个找到的模型，有动手能力的可以看代码修改 `web\__init__.py`。

### 3.2 启动工具箱：
`python demo_toolbox.py -d <datasets_root>`
> 请指定一个可用的数据集文件路径，如果有支持的数据集则会自动加载供调试，也同时会作为手动录制音频的存储目录。

<img width="1042" alt="d48ea37adf3660e657cfb047c10edbc" src="https://user-images.githubusercontent.com/7423248/134275227-c1ddf154-f118-4b77-8949-8c4c7daf25f0.png">

## 文件结构（目标读者：开发者）
├─archived_untest_files 废弃文件
├─encoder encoder模型
│  ├─data_objects
│  └─saved_models 预训练好的模型
├─samples 样例语音
├─synthesizer  synthesizer模型
│  ├─models
│  ├─saved_models 预训练好的模型
│  └─utils 工具类库
├─toolbox 图形化工具箱
├─utils 工具类库
├─vocoder  vocoder模型（目前包含hifi-gan、wavrnn）
│  ├─hifigan
│  ├─saved_models 预训练好的模型
│  └─wavernn
└─web
    ├─api
    │  └─Web端接口
    ├─config
    │  └─ Web端配置文件
    ├─static 前端静态脚本
    │  └─js 
    ├─templates 前端模板
    └─__init__.py Web端入口文件

## 引用及论文
> 该库一开始从仅支持英语的[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) 分叉出来的，鸣谢作者。

| URL | Designation | 标题 | 实现源码 |
| --- | ----------- | ----- | --------------------- |
| [2010.05646](https://arxiv.org/abs/2010.05646) | HiFi-GAN (vocoder)| Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | 本代码库 |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | 本代码库 |

## 常見問題(FQ&A)
#### 1.數據集哪裡下載?
[aidatatang_200zh](http://www.openslr.org/62/)、[magicdata](http://www.openslr.org/68/)、[aishell3](http://www.openslr.org/93/)
> 解壓 aidatatang_200zh 後，還需將 `aidatatang_200zh\corpus\train`下的檔案全選解壓縮

#### 2.`<datasets_root>`是什麼意思?
假如數據集路徑為 `D:\data\aidatatang_200zh`，那麼 `<datasets_root>`就是 `D:\data`

#### 3.訓練模型顯存不足
訓練合成器時：將 `synthesizer/hparams.py`中的batch_size參數調小
```
//調整前
tts_schedule = [(2,  1e-3,  20_000,  12),   # Progressive training schedule
                (2,  5e-4,  40_000,  12),   # (r, lr, step, batch_size)
                (2,  2e-4,  80_000,  12),   #
                (2,  1e-4, 160_000,  12),   # r = reduction factor (# of mel frames
                (2,  3e-5, 320_000,  12),   #     synthesized for each decoder iteration)
                (2,  1e-5, 640_000,  12)],  # lr = learning rate
//調整後
tts_schedule = [(2,  1e-3,  20_000,  8),   # Progressive training schedule
                (2,  5e-4,  40_000,  8),   # (r, lr, step, batch_size)
                (2,  2e-4,  80_000,  8),   #
                (2,  1e-4, 160_000,  8),   # r = reduction factor (# of mel frames
                (2,  3e-5, 320_000,  8),   #     synthesized for each decoder iteration)
                (2,  1e-5, 640_000,  8)],  # lr = learning rate
```

聲碼器-預處理數據集時：將 `synthesizer/hparams.py`中的batch_size參數調小
```
//調整前
### Data Preprocessing
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 16,                  # For vocoder preprocessing and inference.
//調整後
### Data Preprocessing
        max_mel_frames = 900,
        rescale = True,
        rescaling_max = 0.9,
        synthesis_batch_size = 8,                  # For vocoder preprocessing and inference.
```

聲碼器-訓練聲碼器時：將 `vocoder/wavernn/hparams.py`中的batch_size參數調小
```
//調整前
# Training
voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5
voc_pad = 2

//調整後
# Training
voc_batch_size = 6
voc_lr = 1e-4
voc_gen_at_checkpoint = 5
voc_pad =2
```

#### 4.碰到`RuntimeError: Error(s) in loading state_dict for Tacotron: size mismatch for encoder.embedding.weight: copying a param with shape torch.Size([70, 512]) from checkpoint, the shape in current model is torch.Size([75, 512]).`
請參照 issue [#37](https://github.com/babysor/MockingBird/issues/37)

#### 5.如何改善CPU、GPU佔用率?
適情況調整batch_size參數來改善

#### 6.發生 `頁面文件太小，無法完成操作`
請參考這篇[文章](https://blog.csdn.net/qq_17755303/article/details/112564030)，將虛擬內存更改為100G(102400)，例如:档案放置D槽就更改D槽的虚拟内存

#### 7.什么时候算训练完成？
首先一定要出现注意力模型，其次是loss足够低，取决于硬件设备和数据集。拿本人的供参考，我的注意力是在 18k 步之后出现的，并且在 50k 步之后损失变得低于 0.4
![attention_step_20500_sample_1](https://user-images.githubusercontent.com/7423248/128587252-f669f05a-f411-4811-8784-222156ea5e9d.png)

![step-135500-mel-spectrogram_sample_1](https://user-images.githubusercontent.com/7423248/128587255-4945faa0-5517-46ea-b173-928eff999330.png)

