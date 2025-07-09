from utils import my_summary
from basicsr.models.archs.RetinexFormer_arch import QuadPriorFormer

my_summary(QuadPriorFormer(), 256, 256, 3, 1)