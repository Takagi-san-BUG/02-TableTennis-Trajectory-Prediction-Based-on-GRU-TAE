from pathlib import Path

# 所谓“相对路径”，就是相对于当前工作目录的路径，想要动态识别，用__file__，表示当前文件的路径
# 学习使用一个新库Path
# "__file__"无法在jupyter notebook中使用
ROOT_DIR = Path(__file__).parent.parent.parent

VIDEO_DATA_DIR = ROOT_DIR / "data" / "video"  
