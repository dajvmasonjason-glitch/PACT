#!/bin/bash
# 新增实验批量运行示例脚本
#
# 使用方法：
#   chmod +x run_new_experiments.sh
#   ./run_new_experiments.sh

echo "========================================"
echo "新增实验批量运行脚本"
echo "========================================"
echo ""

# 配置
TASK_PAIRS="SUN397:Cars,GTSRB:MNIST,DTD:RESISC45"
USE_CONFIG="--use-config"
SKIP_MISSING="--skip-missing"

# 选项1: 运行所有预定义的任务对
echo "选项1: 运行所有预定义的任务对"
echo "  python -m analysis.run_noise_batch $USE_CONFIG $SKIP_MISSING"
echo "  python -m analysis.run_new_mask_batch $USE_CONFIG $SKIP_MISSING"
echo ""

# 选项2: 运行指定的任务对
echo "选项2: 运行指定的任务对"
echo "  python -m analysis.run_noise_batch --task-pairs \"$TASK_PAIRS\" $USE_CONFIG"
echo "  python -m analysis.run_new_mask_batch --task-pairs \"$TASK_PAIRS\" $USE_CONFIG"
echo ""

# 询问用户选择
read -p "请选择运行模式 (1/2) [默认: 1]: " choice
choice=${choice:-1}

if [ "$choice" == "1" ]; then
    echo ""
    echo "========================================"
    echo "运行所有预定义的任务对"
    echo "========================================"
    echo ""

    # 运行噪声实验
    echo "[1/2] 批量运行噪声实验..."
    python -m analysis.run_noise_batch $USE_CONFIG $SKIP_MISSING

    if [ $? -eq 0 ]; then
        echo "✅ 噪声实验完成"
    else
        echo "❌ 噪声实验失败"
        exit 1
    fi

    echo ""

    # 运行新mask策略实验
    echo "[2/2] 批量运行新mask策略实验..."
    python -m analysis.run_new_mask_batch $USE_CONFIG $SKIP_MISSING

    if [ $? -eq 0 ]; then
        echo "✅ 新mask策略实验完成"
    else
        echo "❌ 新mask策略实验失败"
        exit 1
    fi

elif [ "$choice" == "2" ]; then
    echo ""
    echo "========================================"
    echo "运行指定的任务对: $TASK_PAIRS"
    echo "========================================"
    echo ""

    # 运行噪声实验
    echo "[1/2] 批量运行噪声实验..."
    python -m analysis.run_noise_batch --task-pairs "$TASK_PAIRS" $USE_CONFIG

    if [ $? -eq 0 ]; then
        echo "✅ 噪声实验完成"
    else
        echo "❌ 噪声实验失败"
        exit 1
    fi

    echo ""

    # 运行新mask策略实验
    echo "[2/2] 批量运行新mask策略实验..."
    python -m analysis.run_new_mask_batch --task-pairs "$TASK_PAIRS" $USE_CONFIG

    if [ $? -eq 0 ]; then
        echo "✅ 新mask策略实验完成"
    else
        echo "❌ 新mask策略实验失败"
        exit 1
    fi

else
    echo "无效的选择"
    exit 1
fi

echo ""
echo "========================================"
echo "🎉 所有实验完成！"
echo "========================================"
echo ""
echo "结果保存在: analysis/outputs/{TaskA}-{TaskB}/"
echo "  - noise_only_results.*"
echo "  - new_mask_strategy/new_mask_results.*"
