from vstools import vs, core, initialize_clip, get_y, get_w, depth, join, finalize_clip, set_output, rfs 
from vskernels import SetsuCubic, Lanczos
from vodesfunc import NNEDI_Doubler
from vsaa import fine_aa
from vsscale import SSIM, descale_detail_mask
from vsmasktools import KirschTCanny
from dfttest2 import DFTTest, Backend
from vsdeband import F3kdb, AddNoise
from vsmasktools import FDoG
from vsdehalo import fine_dehalo, dehalo_sigma
from vsdenoise import Prefilter

# Loading Video
src = core.dgdecodenv.DGSource(r'Boku dake ga Inai Machi S01E01 Flashing Before My Eyes 1080p Blu-ray Remux AVC FLAC 2.0-E.N.D.dgi')
src = initialize_clip(src, 16)

# Splitting ED and Main video
clip_ED = src[30713:]
clip = src[:30713]

# Rescale
clip_y = get_y(clip)
nh = 720
nw = get_w(nh)
desc_y = Lanczos(3).descale(clip_y, nw, nh)
upsc_y = NNEDI_Doubler(False).double(desc_y)
resc_y = SetsuCubic.scale(upsc_y, 1920, 1080)

# Noob level AA
f_aa = depth(fine_aa(depth(resc_y, 32)), 16)

# Rescale Mask
reupscaled_clip = Lanczos(3).scale(desc_y, 1920, 1080)
desc_mask = descale_detail_mask(clip_y, reupscaled_clip, thr=0.04, inflate=2, xxpand=(4, 0))

# To protect the original background grain from getting altered/destroyed, applying the rescaling only on line-art.
line_mask = KirschTCanny.edgemask(clip_y, lthr=80 / 255, hthr=100 / 255).std.Maximum().std.Minimum()
apply_line_mask = core.std.MaskedMerge(clip_y, f_aa, line_mask)
apply_desc_mask = core.std.MaskedMerge(apply_line_mask, clip_y, desc_mask)
merge_yuv = join(apply_desc_mask, clip)

# Denoising
sloc = [0.0, 0.4, 0.2, 0.8, 0.5, 1, 0.8, 1.2, 1.0, 1.5]
denoise = DFTTest(merge_yuv, ftype=1, slocation=sloc, backend=Backend.CPU())

# Dehaloing
dehalo = dehalo_sigma(denoise, pre_ss=2, sigma=2.5, blur_func=Prefilter.GAUSSBLUR2)
dehalo = fine_dehalo.mask(denoise, dehalo)

# Noob level Debanding
mild_deband = F3kdb.deband(dehalo, 16, [30*4, 16*4, 16*4])
heavy_deband = F3kdb.deband(dehalo, 16, [70*4, 16*4, 16*4])
frame_ranges = [(5675, 6261), (6683, 7138), (7585,7907), (8439,8706)]
debanded = rfs(mild_deband, heavy_deband, frame_ranges)

# Edge Mask to protect the edges
mask = FDoG.edgemask(clip_y, 0.18, 0.18).std.Maximum().std.Minimum()
apply_mask = core.std.MaskedMerge(debanded, merge_yuv, mask)

# Add Noise
grain = AddNoise.GAUSS.grain(apply_mask, (0.3, 0), luma_scaling=8)

# Merging ED 
merged_clip = grain + clip_ED

final = finalize_clip(merged_clip, 10)

set_output(final)
