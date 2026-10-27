# Versioned adapter specifications

本目录只保存不含真实像素、不含本地 root、且已经冻结的显式 adapter spec。真实数据仍必须放在 Git
仓库外；spec 中的 published image key 只作为 source-image identity，不得推断 patient、eye、follow-up
或 intervention-pair identity。

`idrid-he-ex-s1-v1.json` 与 `.summary.json` 由 `python -m retinaentail.data.idrid_freeze` 从明确提供的
IDRiD root 生成。
它只声明 HE/EX 任务目录：缺任一 mask 的图像保持 `UNKNOWN` 并排除于 complete-label cohort，绝不转成
阴性。官方 test 固定为 `LOCKED_TEST`；官方 train 以 `SHA256(source_image_id)` 前 8 字节的大端整数
对 5 取模，桶 0 为 `CALIBRATION`，其余为 `TRAIN`。

S1 对 exact release 实测的 160 个 HE/EX TIFF 执行受控的只读内存归一化：159 个为 palette index
`0 -> (0,0,0)`、`1 -> (255,0,0)`，1 个为 opaque RGBA black/red。审计仅接受这两个精确二值编码；
任何其他 palette、透明度或颜色继续 fail closed，且不会把转换结果写回数据 root。
