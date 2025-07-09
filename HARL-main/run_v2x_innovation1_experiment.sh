#!/bin/bash

# V2X第一个创新点验证实验一键运行脚本
# 
# 使用方法:
# bash run_v2x_innovation1_experiment.sh quick    # 快速验证
# bash run_v2x_innovation1_experiment.sh full     # 完整实验

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 检查必要的Python包
    python -c "import torch, numpy, matplotlib" 2>/dev/null || {
        print_error "缺少必要的Python包: torch, numpy, matplotlib"
        print_info "请运行: pip install torch numpy matplotlib seaborn pandas"
        exit 1
    }
    
    print_success "Python环境检查通过"
}

# 检查HARL环境
check_harl_env() {
    print_info "检查HARL环境..."
    
    if [ ! -d "HARL-main" ]; then
        print_error "HARL-main目录不存在"
        exit 1
    fi
    
    if [ ! -f "HARL-main/harl/configs/envs_cfgs/v2x.yaml" ]; then
        print_error "V2X配置文件不存在"
        exit 1
    fi
    
    print_success "HARL环境检查通过"
}

# 创建实验目录
setup_experiment_dir() {
    print_info "设置实验目录..."
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_DIR="v2x_innovation1_results_${TIMESTAMP}"
    
    mkdir -p "$EXPERIMENT_DIR"
    mkdir -p "$EXPERIMENT_DIR/logs"
    mkdir -p "$EXPERIMENT_DIR/models"
    mkdir -p "$EXPERIMENT_DIR/figures"
    
    print_success "实验目录创建完成: $EXPERIMENT_DIR"
}

# 运行快速验证实验
run_quick_experiment() {
    print_info "开始快速验证实验..."
    
    cd HARL-main
    
    # 运行实验
    python experiments/run_v2x_experiment.py --mode quick 2>&1 | tee "../${EXPERIMENT_DIR}/quick_experiment.log"
    
    if [ $? -eq 0 ]; then
        print_success "快速验证实验完成"
    else
        print_error "快速验证实验失败"
        return 1
    fi
    
    cd ..
}

# 运行完整实验
run_full_experiment() {
    print_info "开始完整验证实验..."
    
    cd HARL-main
    
    # 运行实验
    python experiments/run_v2x_experiment.py --mode full 2>&1 | tee "../${EXPERIMENT_DIR}/full_experiment.log"
    
    if [ $? -eq 0 ]; then
        print_success "完整验证实验完成"
    else
        print_error "完整验证实验失败"
        return 1
    fi
    
    cd ..
}

# 分析实验结果
analyze_results() {
    print_info "分析实验结果..."
    
    cd HARL-main
    
    # 运行结果分析
    python experiments/analyze_results.py --result_dir "../${EXPERIMENT_DIR}" 2>&1 | tee "../${EXPERIMENT_DIR}/analysis.log"
    
    if [ $? -eq 0 ]; then
        print_success "结果分析完成"
    else
        print_warning "结果分析可能不完整，请检查日志"
    fi
    
    cd ..
}

# 生成实验报告
generate_report() {
    print_info "生成实验报告..."
    
    REPORT_FILE="${EXPERIMENT_DIR}/experiment_report.md"
    
    cat > "$REPORT_FILE" << EOF
# V2X第一个创新点验证实验报告

## 实验信息
- 实验时间: $(date)
- 实验类型: $EXPERIMENT_MODE
- 实验目录: $EXPERIMENT_DIR

## 实验目的
验证"动态上下文感知状态表征"创新点的效果，包括：
1. Transformer编码器对历史序列的建模能力
2. 对比学习对状态表征质量的提升
3. 两者结合对整体性能的改进

## 实验配置
- 基线: 标准HASAC算法
- 对比1: HASAC + Transformer编码器
- 对比2: HASAC + Transformer + 对比学习（完整创新点）

## 实验场景
- 低动态场景: 车辆低速移动，网络相对稳定
- 中等动态场景: 标准V2X环境
- 高动态场景: 高速移动，网络快速变化

## 结果文件
- 学习曲线图: figures/learning_curves.png
- 性能对比图: figures/performance_comparison.png
- 对比学习损失图: figures/contrastive_learning_loss.png
- 详细日志: *.log
- 摘要报告: summary_report.txt

## 使用说明
1. 查看 summary_report.txt 了解总体结果
2. 查看 figures/ 目录下的图表了解详细对比
3. 查看 *.log 文件了解训练过程

## 下一步
根据实验结果：
- 如果创新点效果显著，可以进行更大规模实验
- 如果效果一般，需要调优超参数或网络结构
- 可以尝试不同的场景配置验证泛化性能
EOF

    print_success "实验报告已生成: $REPORT_FILE"
}

# 主函数
main() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "          V2X第一个创新点验证实验自动化脚本"
    echo "================================================================"
    echo -e "${NC}"
    
    # 检查参数
    EXPERIMENT_MODE=${1:-"quick"}
    
    if [[ "$EXPERIMENT_MODE" != "quick" && "$EXPERIMENT_MODE" != "full" ]]; then
        print_error "无效参数: $EXPERIMENT_MODE"
        echo "使用方法: $0 [quick|full]"
        exit 1
    fi
    
    print_info "实验模式: $EXPERIMENT_MODE"
    
    # 环境检查
    check_python_env
    check_harl_env
    
    # 设置实验
    setup_experiment_dir
    
    # 运行实验
    if [ "$EXPERIMENT_MODE" = "quick" ]; then
        run_quick_experiment
    else
        run_full_experiment
    fi
    
    # 分析结果
    analyze_results
    
    # 生成报告
    generate_report
    
    echo -e "${GREEN}"
    echo "================================================================"
    echo "                        实验完成!"
    echo "================================================================"
    echo -e "${NC}"
    
    print_success "所有结果保存在: $EXPERIMENT_DIR"
    print_info "请查看 $EXPERIMENT_DIR/summary_report.txt 了解实验结果"
    print_info "请查看 $EXPERIMENT_DIR/figures/ 目录了解可视化结果"
    
    # 显示关键结果
    if [ -f "$EXPERIMENT_DIR/summary_report.txt" ]; then
        echo ""
        print_info "实验摘要:"
        echo "================================================================"
        cat "$EXPERIMENT_DIR/summary_report.txt"
        echo "================================================================"
    fi
}

# 运行主函数
main "$@" 