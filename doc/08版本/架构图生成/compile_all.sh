#!/bin/bash
# 批量编译所有架构图
# 使用方法: ./compile_all.sh

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  论文架构图批量编译脚本"
echo "=================================================="
echo ""

# 检查是否安装了 pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "❌ 错误: 未安装 pdflatex"
    echo ""
    echo "解决方法："
    echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-extra"
    echo "  macOS: brew install --cask mactex"
    echo ""
    echo "或者使用在线编译器："
    echo "  Overleaf: https://www.overleaf.com/"
    exit 1
fi

# 定义要编译的文件
files=(
    "shape_architecture.tex"
    "system_architecture.tex"
    "multimodal_pipeline.tex"
)

# 编译每个文件
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "📄 编译 $file ..."
        pdflatex -interaction=nonstopmode "$file" > /dev/null 2>&1

        if [ $? -eq 0 ]; then
            echo "   ✅ 成功生成 ${file%.tex}.pdf"
        else
            echo "   ❌ 编译失败，请查看 ${file%.tex}.log"
        fi
    else
        echo "   ⚠️  文件不存在: $file"
    fi
done

echo ""
echo "=================================================="
echo "  编译完成！"
echo "=================================================="
echo ""

# 清理辅助文件
echo "🧹 清理辅助文件..."
rm -f *.aux *.log *.out *.synctex.gz
echo "   ✅ 清理完成"

echo ""
echo "生成的PDF文件："
ls -lh *.pdf 2>/dev/null || echo "   未生成PDF文件"

echo ""
echo "下一步："
echo "  1. 查看PDF: evince shape_architecture.pdf"
echo "  2. 转换PNG: ./convert_to_png.sh"
echo "  3. 插入论文: 参考 README.md"
