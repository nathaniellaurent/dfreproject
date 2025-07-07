import sys
sys.path.append('src')  # Adds the parent directory to the Python path
print(sys.path)
from astropy.io import fits
from astropy.wcs import WCS
from dfreproject import calculate_reprojection
from sunpy.data.sample import AIA_193_JUN2012, STEREO_A_195_JUN2012
import matplotlib.pyplot as plt
import torch
from astropy.io.fits import PrimaryHDU
from dfreproject import TensorHDU

from sunpy.map import Map
import astropy.units as u


# Load source and target images
source_hdu = fits.open(AIA_193_JUN2012)[1]
target_hdu = fits.open(STEREO_A_195_JUN2012)[1]
source_hdu = PrimaryHDU(source_hdu.data, header=source_hdu.header)

aia_map = Map(AIA_193_JUN2012)
stereo_map = Map(STEREO_A_195_JUN2012)

# --- SunPy Map reproject_to for comparison ---
aia_map.wcs.wcs.aux.rsun_ref = 6.957e8  # Set the solar radius for AIA
stereo_map.wcs.wcs.aux.rsun_ref = 6.957e8  #
reprojected_sunpy_map = aia_map.reproject_to(stereo_map.wcs)
reprojected_sunpy = reprojected_sunpy_map.data

# source_hdu = (torch.tensor(source_hdu.data, requires_grad=True), source_hdu.header)

source_hdu = TensorHDU(torch.tensor(source_hdu.data, requires_grad=True), source_hdu.header)
alt_tensor = source_hdu.tensor 
alt_header = source_hdu.header
target_hdu = PrimaryHDU(target_hdu.data, header=target_hdu.header)

source_wcs = WCS(source_hdu.header)
target_wcs = WCS(target_hdu.header)
source_hdu.header['RSUN_REF'] = 6.957e8  # Set solar radius for source
target_wcs.wcs.aux.rsun_ref = 6.957e8  #
# Use the downscale_and_update_fits function defined above
# downscale_size = 1024
# source_hdu, target_hdu, target_wcs = downscale_and_update_fits(source_hdu, target_hdu, target_wcs, downscale_size=downscale_size)
print(type(source_hdu))

# Extract HGLT_OBS (heliographic latitude of observer) from WCS if present
print("Lat", source_wcs.wcs.aux.hglt_obs)
print("Lon", source_wcs.wcs.aux.hgln_obs)
# print("HGLT_OBS from WCS:", hglt_obs)


reprojected = calculate_reprojection(
    source_hdus = source_hdu,
    target_wcs=target_wcs,
    shape_out=target_hdu.data.shape,
    order='bilinear',
    requires_grad=True,
)

print(reprojected.shape)  # Should match target_hdu.data.shape


print(reprojected.requires_grad)  # Should be True if requires_grad=True was passed
reprojected.sum().backward()
print(source_hdu.tensor.grad)
# Plot with a diverging colormap centered at zero for gradients
import numpy as np

grad = source_hdu.tensor.grad.cpu().numpy()
# Use percentiles to avoid outlier-dominated color scaling
vmax = np.nanpercentile(np.abs(grad), 99)
if vmax == 0 or np.isnan(vmax):
    vmax = 1e-8  # fallback to avoid division by zero

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(source_hdu.data, cmap='magma')
axes[0].set_title('Source (AIA)')
axes[1].imshow(target_hdu.data, cmap='viridis')
axes[1].set_title('Target (STEREO)')
axes[2].imshow(reprojected.detach(), cmap='magma')
axes[2].set_title('Reprojected (AIA to STEREO)')
im = axes[3].imshow(grad, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[3].set_title('Backward Output (Grad)')
for ax in axes:
    ax.axis('off')
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
plt.tight_layout()

# Overlay the target (STEREO) and reprojected (AIA to STEREO) images
fig_overlay, ax_overlay = plt.subplots(figsize=(8, 8))
# Show STEREO in green channel
ax_overlay.imshow(reprojected_sunpy, cmap='Greens', alpha=0.7, label='STEREO')
# Show reprojected AIA in red channel
im_overlay = ax_overlay.imshow(reprojected.detach(), cmap='Reds', alpha=0.5, label='Reprojected AIA')
ax_overlay.set_title('Overlay: Sunpy (green) + Reprojected AIA (red)')
ax_overlay.axis('off')
fig_overlay.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)

# Display the difference between SunPy reprojection and our reprojection
fig_diff, ax_diff = plt.subplots(figsize=(8, 8))
diff = reprojected.detach().cpu().numpy() - reprojected_sunpy
im_diff = ax_diff.imshow(diff, cmap='coolwarm')
ax_diff.set_title('Difference: Ours - SunPy')
ax_diff.axis('off')
fig_diff.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

# Compute relative error in percent
error = 100 * (reprojected.detach().cpu().numpy() - reprojected_sunpy) / (reprojected_sunpy + 1e-8)
print("All pixels are NaN in error:", np.isnan(error).all())

fig_error, ax_error = plt.subplots(figsize=(8, 8))
im_error = ax_error.imshow(error, origin='lower', cmap='seismic', vmin=-3, vmax=3)
ax_error.set_title('Relative Error (%)')
ax_error.axis('off')
fig_error.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)


# Plot just the SunPy reprojected output
fig_compare, axes_compare = plt.subplots(1, 2, figsize=(16, 8))

# SunPy reprojected output
im_sunpy = axes_compare[0].imshow(reprojected_sunpy, cmap='Greens')
axes_compare[0].set_title('SunPy Reprojected Output')
axes_compare[0].axis('off')
fig_compare.colorbar(im_sunpy, ax=axes_compare[0], fraction=0.046, pad=0.04)

# Your reprojection output
im_ours = axes_compare[1].imshow(reprojected.detach(), cmap='Reds')
axes_compare[1].set_title('Our Reprojected Output')
axes_compare[1].axis('off')
fig_compare.colorbar(im_ours, ax=axes_compare[1], fraction=0.046, pad=0.04)


plt.show()