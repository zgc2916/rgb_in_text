# InfMAE 在 RefFLIR 上冲击 76%：截至 2026-09-27 的实验记录

> 2026-10-03 新进度：GPU3/6已开始红外前端RGB教师置信度加权实验（12轮、lr5e-6、seed13），之后顺序运行12轮等权匹配对照及120轮原CLIP-TIR/IAFv3公共初始化同配对照。只训练459.5万红外前端参数，真实优化步确认902个冻结张量保持原值，初始验证仍为457/608、329/608、63.472%。弱光和small标签初始Acc@0.5分别为76/114和210/350。尚未获得新提升结论，测试集封存。配置、梯度检查与运行状态见`output_infmae_a5f2_semantic/2026-10-03_seed13_frontend12_lr5e6/`，完整分析见会话记录末节。

## 目标与可比口径

目标是 RefFLIR 验证集 Acc@0.5 **至少 463/608 = 76.151%**，并同时报告同一权重的 Acc@0.7 和 mIoU。A5-F2 原实验 epoch 94 为 **457/608 = 75.164%**、Acc@0.7 **329/608 = 54.112%**、mIoU **63.472%**。新实验均保持 RGB CLIP + TIR InfMAE、LAVS、IAFv3 和 VL 定位主干；不加载 GQR 权重。

## 已做快速排查

从 A5-F2 epoch 94 权重在单卡上逐样本评估，原图得到 456/608，比原两卡验证日志少 1 个边界样本。随后用严格加载的同一权重在两卡训练启动时复核，得到原日志的 457/608。因此新训练结果按项目原有两卡验证函数判断。

| 单卡探索 | Acc@0.5 | Acc@0.7 | mIoU | 判断 |
| --- | ---: | ---: | ---: | --- |
| 原图 | 456/608 | 329/608 | 63.486% | 逐样本分析参照 |
| 原图 + 未改文本的水平镜像等权平均 | 360/608 | 237/608 | 49.948% | 方位词语义不匹配，弃用 |
| 原图 + mask 框等权平均 | 454/608 | 325/608 | 63.808% | Acc@0.5 下降，弃用 |
| epoch 94 与 119 权重等权平均 | 457/608 | 328/608 | 63.699% | 没有提高 Acc@0.5 |

逐样本文件在 output_infmae_analysis/，不进入公开 Git。按验证变换后的 GT 框面积统计，面积低于画面 0.2% 的 141 个极小目标只有 47 个正确（33.3%）；面积 1% 到 5% 的 175 个目标有 167 个正确（95.4%）。这是下一步聚焦 F2 局部细节的直接依据。该分组是错误诊断，不用 GT 面积控制推理分支。

## 对照一：A5-F2 最佳权重低学习率续训

训练入口为 script_train/infmae_experiments/continue_a5f2.py。它从原 checkpoint 提取模型权重到临时 model-only checkpoint，再通过项目原有 --resume 严格加载模型，重置优化器和余弦调度器。原始模型结构、数据加载和训练循环不变。

- 初始化：原 A5-F2 epoch 94 最佳权重；两卡启动验证复核 457/608。
- 训练：2 张 4090、seed 13、每卡 batch 6、累积 8、有效 batch 96、InfoNCE 候选数 12、16 epoch、AdamW、基础及对齐头学习率 1e-5。
- 目标损失：TIR–Text 0.05、RGB→TIR 0.02，续训从 epoch 0 起以完整权重启用。
- 输出：output_infmae_a5f2_continue/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_13_ep94_restart16_lr1e5/。

训练已完成。最佳为 epoch 6：Acc@0.5 **464/608 = 76.316%**、Acc@0.7 **319/608 = 52.467%**、mIoU **63.906%**。epoch 8 的 mIoU 略高（64.038%），但 Acc@0.5 是 462/608，因此目标 checkpoint 选 epoch 6。

## 对照二：F2 峰值残差变体

新增 InfMAEA5BridgeFTF2Peak，仅改变 InfMAE F2 的 28×28 到 14×14 前端接口。在原双线性下采样旁引入 2×2 最大池化的局部响应，使用 **768 通道零初始化门控** 混合两者。门控为零时前向与 A5-F2 完全相同，避免把新结构随机初始化误差当成性能变化。F2/F3 LoRA、语义桥、LAVS、IAFv3、VL Transformer 和损失保持不变。

模块测试验证零门控输出与原适配器逐元素相同且门控有非零梯度；真实 checkpoint 加入零门控后严格加载成功。此变体应与对照一使用同一源权重、seed、batch、候选数、学习率和续训轮数；只有跑完才能判定是否有效。

准备命令：

~~~bash
python -m script_train.infmae_experiments.continue_a5f2 \
  --prepare --peak \
  --source output_infmae_a5_f2full/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_13_full120_g2g4/best_checkpoint.pth \
  --warmstart /tmp/infmae_a5f2_peak_ep94_model_only.pth
~~~

训练命令：

~~~bash
CUDA_VISIBLE_DEVICES=4,5 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m script_train.infmae_experiments.continue_a5f2 --peak \
  --source output_infmae_a5_f2full/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_13_full120_g2g4/best_checkpoint.pth \
  --warmstart /tmp/infmae_a5f2_peak_ep94_model_only.pth \
  --output_dir /tmp/infmae_a5f2_peak/InfMAEA5BridgeFTF2Peak_224/rgbtvg_flir/seed_13_ep94_restart16_lr1e5 \
  --epochs 16 --lr 1e-5
~~~

对照二已完成 16 轮，输出保存在 `/tmp/infmae_a5f2_peak/InfMAEA5BridgeFTF2Peak_224/rgbtvg_flir/seed_13_ep94_restart16_lr1e5/`。最高 Acc@0.5 为 epoch 6/14 的 **459/608 = 75.493%**；epoch 14 同权重 Acc@0.7 **323/608 = 53.125%**、mIoU **63.625%**。最高 mIoU 出现在 epoch 8，为 **63.681%**，但 Acc@0.5 只有 458/608。该变体没有超过 A5-F2 对照，不建议作为主配置。

测试：pytest -q tests/test_infmae_modules.py tests/test_iafv3_equivalence.py，共 25 项通过。

## 多 seed 验证与测试集结果

| 方案 | Seed | 最高 Acc@0.5 | 同 epoch Acc@0.7 | 同 epoch mIoU | 全程最高 mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| A5-F2 低学习率续训 | 13 | epoch 6: **464/608 (76.316%)** | 319/608 (52.467%) | 63.906% | epoch 8: 64.038% |
| A5-F2 低学习率续训 | 14 | epoch 2: 458/608 (75.329%) | 327/608 (53.783%) | 63.765% | epoch 6: 63.875% |
| F2 峰值残差 | 13 | epoch 6/14: 459/608 (75.493%) | epoch 14: 323/608 (53.125%) | 63.625% | epoch 8: 63.681% |

A5-F2 seed 14 的日志和权重在 `/tmp/infmae_a5f2_continue_seed14/InfMAEA5BridgeFTF2_224/rgbtvg_flir/seed_14_ep94_restart16_lr1e5/`。seed 13 的 464/608 是单次越线结果；seed 14 最好只有 458/608，差 6 个样本，说明当前 76% 结果对随机种子敏感。F2 残差对照也未越线，不建议作为主配置。

仅对按验证集 Acc@0.5 选出的 seed 13 epoch 6 权重评估了一次测试集：Acc@0.5 **1483/2104 = 70.485%**、Acc@0.7 **1058/2104 = 50.285%**、mIoU **59.083%**。结果文件是 `output_infmae_analysis/a5f2_continue_epoch6_test.json`。该测试 Acc@0.5 低于基线论文报告的 72.43%；当前可以说验证集单次峰值超过 76%，不能说测试集超过 76% 或者跨 seed 稳定超过 76%。seed 14 与 F2 残差方案没有查看测试集，模型选择只依据验证集。

## éªè¯éæéå¹³åä¸ GIoU å¯¹ç§

- å°åå§ A5-F2 epoch 94 ä¸ seed 13 ä½å­¦ä¹ çç»­è®­ epoch 6 çå¨é¨æµ®ç¹åæ°ç­æå¹³åï¼ç»æãæ¨çä»£ç åæ°æ®åä¸åãåå¡éæ ·æ¬è¯ä¼°ä¸º 463/608ï¼éåä½¿ç¨é¡¹ç®åæä¸¤å¡éªè¯å½æ°å¤æ ¸ä»ä¸º **463/608 = 76.151% Acc@0.5**ãè¿æ¯å½åå¯ä¸ç»æ­£å¼éªè¯è¾¾å°ç®æ çº¿ç InfMAE åéãè¯¥åéåªç¨äºéªè¯ééæ©ï¼å°æªè¯ä¼°æµè¯éã
- è¯¥å¹³åå¹¶ä¸è½è¯æè·¨ seed ç¨³å®ï¼seed 13 ä¸ seed 14 ç»­è®­æ¨¡åç­æå¹³ååªæ 462/608ï¼åå§æ¨¡åä¸ seed 14 çç­æå¹³åä¸º 460/608ãå æ­¤ä¸è½æä¸ä¸ª seed çåæ°å¹³ååæç¨³å®çåæ¨¡åç»è®ºã
- ä¸ºè§£å³ä½å­¦ä¹ çç»­è®­ä¸­ Acc@0.7 ä¸éçé®é¢ï¼æ°å¢å¯éç box_l1_weight å box_giou_weightï¼é»è®¤ä»æ¯ 2.0/2.0ï¼ä» seed 13 ä» epoch 94 ä»¥ 2.0/2.5 è¿è¡å¯¹ç§ç»­è®­ãå° epoch 6 çæå¥½ Acc@0.5 ä¸º 460/608 = 75.658%ï¼Acc@0.7 ä¸º 319/608 = 52.467%ï¼mIoU ä¸º 63.763%ãå®æ²¡æè¶è¿é»è®¤ç»­è®­ç 464/608ï¼ä¸ä»ä½äºåæ¨¡åç Acc@0.7 329/608ï¼å æ­¤å¨ epoch 6 åæååæ­¢ï¼ä¸è¿å¥ seed 14ãæ¥å¿å checkpoint ä¿çå¨ /tmp/infmae_a5f2_giou25/ã
- ä»£ç åå½ï¼pytest -q tests/test_infmae_modules.py tests/test_iafv3_equivalence.py éè¿ 25 é¡¹ã

### åç»­ä¼åçº§

1. å»ç»æµè¯éï¼ç»ä¸ä»¥ Acc@0.5ãAcc@0.7ãmIoU ååç§/åºæ¯/æ¥æ¤åº¦åç»æ¥åç­ééªè¯åéã
2. ä¸åéå¤ F2 å³°å¼ãGQRãTCRCãGIoU å ææåè°åº¦ç»­è®­ãä¸ä¸é¡¹ç»æå®éªåºå¨ A5-F2 å° LAVS/IAFv3 çæ¥å£ä¸­ï¼åªå¼å¥é¶åå§åä¸åçº¦æçæ¨¡æå¯é æ§æ°å¨ï¼å¹¶ä»å epoch 94 æéååéæ¹ãå seed å¯¹ç§ã
3. åªæéç½®å¨ seed 13 å seed 14 é½è¶è¿ 463/608 ä¸ä¸éä½ Acc@0.7 åï¼æå¯¹éå®æéæ§è¡ä¸æ¬¡å»ç»æµè¯éè¯ä¼°ã
