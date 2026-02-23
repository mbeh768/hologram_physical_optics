import numpy as np
import tifffile
import matplotlib.pyplot as plt

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

# convert to magnitude and phase
G_mag = np.abs(G)
G_phase = np.angle(G)

# normalize magnitudes at each pixel
G_mag_norm = G_mag / G_mag.max()

# select cell size 
cell_size = 8

# normalize magnitude
G_mag_norm_quant = np.round( cell_size * G_mag_norm )

# plot quantized magnitude histogram
plt.hist(G_mag_norm_quant.astype(int).flatten(), bins=cell_size, edgecolor='black')
plt.title("Histogram of quantized magnitude")
plt.show()

# quantization with error diffusion
error_diffusion = True

if error_diffusion:
    G_phase_quant = np.zeros_like(G_phase)
    error = np.complex64(0.0)
    for i, phi in enumerate(G_phase.flatten()):
        phi_with_error = np.angle( np.exp(1j * phi) + error )
        bin_index = np.floor( cell_size * (phi_with_error + np.pi) / (2 * np.pi) ) # [0, 2]
        G_phase_quant.flat[i] = bin_index                                    # write quantized phase to array

        # convert quantized phase into radians
        phi_quant_rads = (bin_index + 0.5) * (2 * np.pi / cell_size) - np.pi

        # calculate complex error
        error = np.exp(1j * phi) - np.exp(1j * phi_quant_rads)

else:
    G_phase_quant = np.floor( cell_size * (G_phase + np.pi) / (2 * np.pi) ) # [0, 2]

# plot quantized phase histogram
plt.hist(G_phase_quant.astype(int).flatten(), bins=cell_size, edgecolor='black')
plt.title("Histogram of quantized phase")
plt.show()

# for each pixel, create the appropriate cell_size x cell_size array
canvas = np.zeros( (cell_size * image.shape[0], cell_size * image.shape[1]), dtype = np.uint8 )
for y_G, y in enumerate(range(0, canvas.shape[0], cell_size)):  
    for x_G, x in enumerate(range(0, canvas.shape[1], cell_size)):
        # find magnitude and phase values for the current pixel
        magnitude = int(G_mag_norm_quant[y_G, x_G])
        phase = int(G_phase_quant[y_G, x_G])

        # write the cell array
        cell = np.zeros( (cell_size, cell_size) )

        # fill vertical line at appropriate coord
        cell[:magnitude, phase] = 1 

        # show the first cell produced
        if y_G == 0 and x_G == 0:
            plt.imshow(cell)
            plt.title("First cell produced")
            plt.show()

        # add cell to the canvas
        canvas[y:y+cell_size, x:x+cell_size] = cell

# show the aperture
plt.imshow(canvas * 255)
plt.title("Aperture")
plt.show()

# take fourier transform of the aperture to preview image to be created
canvas_fft = np.fft.fft2(canvas)
canvas_fft_shift = np.fft.fftshift(canvas_fft)

# Compute magnitude of fft
fft_mag = np.log1p(np.abs(canvas_fft_shift))
low, high = np.percentile(fft_mag, [0, 100])        # optionally clip values
fft_mag_clipped = np.clip(fft_mag, low, high)

# display fourier transform of aperture
plt.imshow(fft_mag_clipped, cmap='gray')
plt.title("Fourier Transform of Aperture")
plt.show()