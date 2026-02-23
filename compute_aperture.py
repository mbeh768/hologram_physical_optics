import numpy as np
import tifffile
import matplotlib.pyplot as plt
import tifffile

image_path = 'tiny_image.tiff'

# import image [0-1]
with tifffile.TiffFile(image_path) as f:
    image = f.asarray()

# add random phase to image
random_phase = np.random.uniform(-np.pi, np.pi, size=image.shape)

# combine phase with image
g = image * np.exp( 1j * random_phase )

# compute fourier transform
G = np.fft.fft2(g)
G = np.fft.fftshift(G) # center zero-frequency component at origin

# convert to magnitude and phase in order to normalize magnitude
G_mag = np.abs(G)
G_phase = np.angle(G)

# normalize magnitudes at each pixel
G_mag_norm = G_mag / G_mag.max()

# select cell size 
cell_size = 8

# select line width
line_width = 3

# quantization with error diffusion
error_diffusion = True

if error_diffusion:
    G_phase_quant = np.zeros_like(G_phase)
    G_mag_norm_quant = np.zeros_like(G_mag_norm)
    error = np.complex64(0.0)
    for i, (phi, mag) in enumerate(zip(G_phase.flatten(), G_mag_norm.flatten())):
        
        # magnitude error diffusion
        g_with_error = mag * np.exp(1j * phi) + error

        # conver to magnitude and phase
        mag_with_error = np.abs(g_with_error)
        phi_with_error = np.angle(g_with_error)

        # quantize magnitude and phase
        mag_quant = np.round(cell_size * mag_with_error)
        phi_quant = np.floor(cell_size * (phi_with_error + np.pi) / (2 * np.pi))

        # append to array
        G_mag_norm_quant.flat[i] = mag_quant
        G_phase_quant.flat[i] = phi_quant

        # convert quantized values back to complex and compute error
        mag_quant_norm = mag_quant / cell_size
        phi_quant_rads = (phi_quant + 0.5) * (2 * np.pi / cell_size) - np.pi
        g_quant = mag_quant_norm * np.exp(1j * phi_quant_rads)

        # compute complex error
        error = g_with_error - g_quant
else:
    G_mag_norm_quant = np.round( cell_size * G_mag_norm )
    G_phase_quant = np.floor( cell_size * (G_phase + np.pi) / (2 * np.pi) ) # [0, 2]

# plot quantized phase histogram
plt.hist(G_phase_quant.astype(int).flatten(), bins=cell_size, edgecolor='black')
plt.title("Histogram of quantized phase")
plt.show()

# plot quantized magnitude histogram
plt.hist(G_mag_norm_quant.astype(int).flatten(), bins=cell_size, edgecolor='black')
plt.title("Histogram of quantized magnitude")
plt.show()

# for each pixel, create the appropriate cell_size x cell_size array
aperture = np.zeros( (cell_size * image.shape[0], cell_size * image.shape[1]), dtype = np.uint8 )
for y_G, y in enumerate(range(0, aperture.shape[0], cell_size)):  
    for x_G, x in enumerate(range(0, aperture.shape[1], cell_size)):
        # find magnitude and phase values for the current pixel
        magnitude_bin = int(G_mag_norm_quant[y_G, x_G])
        phase_bin = int(G_phase_quant[y_G, x_G])

        # write the cell array
        cell = np.zeros( (cell_size, cell_size) )

        # fill vertical line at appropriate coord (with appropriate width, wrapping if too great a value)
        bins_to_write = []
        start = int(phase_bin - np.floor(line_width/2))
        for i in range(line_width):
            adj_bin = (start + i) % (cell_size)                 # wrap
            bins_to_write.append(adj_bin)

        cell[:magnitude_bin, bins_to_write] = 1 

        # show the first cell produced
        if y_G == 0 and x_G == 0:
            plt.imshow(cell)
            plt.title("First cell produced")
            plt.show()

        # add cell to the canvas
        aperture[y:y+cell_size, x:x+cell_size] = cell

# show the aperture
plt.imshow(aperture * 255)
plt.title("Aperture")
plt.show()

tifffile.imwrite('aperture.tiff', aperture * 255)

# take fourier transform of the aperture to preview image to be created
aperture_fft = np.fft.fft2(aperture)
aperture_fft_shift = np.fft.fftshift(aperture_fft)

# Compute magnitude of fft
fft_mag = np.log1p(np.abs(aperture_fft_shift))
low, high = np.percentile(fft_mag, [0, 100])        # optionally clip values
fft_mag_clipped = np.clip(fft_mag, low, high)

# display fourier transform of aperture
plt.imshow(fft_mag_clipped, cmap='gray')
plt.title("Fourier Transform of Aperture")
plt.show()