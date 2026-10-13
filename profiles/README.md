# 个人视线校准

校准文件由本机摄像头、个人坐姿及显示窗口共同决定，`*.json` 仅保存在本地，不随仓库提交。
首次使用或换摄像头、坐姿、窗口模式后，在实际 demo 内校准：

```bash
./run_demo.sh --camera 0 --calibrate profiles/demo_user.json
```

校准通过后自动保存并继续演示；之后加载：

```bash
./run_demo.sh --camera 0 --calibration-file profiles/demo_user.json
```

实时窗口按 `c` 可重新校准。失败的校准不会覆盖原文件，详见 [demo 使用说明](../docs/DEMO.md)。
