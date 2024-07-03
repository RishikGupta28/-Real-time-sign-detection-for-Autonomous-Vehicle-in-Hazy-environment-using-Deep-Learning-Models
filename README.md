# Real-time-Sign-Detection-for-Autonomous-Vehicles-in-Hazy-environment


### Project: Dehazing of Images using Dark Channel Prior

This repository contains the implementation of a dehazing algorithm based on the Dark Channel Prior. The algorithm enhances the visibility of images taken in hazy conditions by removing the haze effect.

### Dataset Details

This implementation can be used on any image dataset requiring dehazing. The input images are expected to be in a standard format such as JPEG or PNG.

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `opencv-python`: For image processing and manipulation.
  - `numpy`: For numerical computations.

### Algorithm and Approach

1. **Dark Channel Prior**:
    - Calculate the dark channel of the image, which represents the minimum intensity in a local patch.

2. **Atmospheric Light Estimation**:
    - Estimate the atmospheric light using the brightest pixels in the dark channel.

3. **Transmission Estimation**:
    - Estimate the transmission map, which indicates the portion of light that is not scattered and reaches the camera.

4. **Guided Filtering**:
    - Refine the transmission map using a guided filter to preserve edge details.

5. **Image Recovery**:
    - Recover the dehazed image using the transmission map and atmospheric light.

### Usage

To use the dehazing algorithm, you need to run the provided Python script with your input image. The output will be a dehazed version of the input image.

### Example Code Snippets

#### Importing Required Libraries
```python
import cv2
import math
import numpy as np
```

#### Dark Channel Calculation
```python
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark
```

#### Atmospheric Light Estimation
```python
def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A
```

#### Transmission Estimation
```python
def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission
```

#### Guided Filtering
```python
def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q
```

#### Image Recovery
```python
def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res
```

### Full Dehazing Function
```python
def dehaze_image(img_path):
    src = cv2.imread(img_path)
    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    cv2.imwrite("./image/J.png", J * 255)
    img2 = cv2.imread("./image/J.png")
    return img2
```

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `opencv-python`: For image processing and manipulation.
  - `numpy`: For numerical computations.
  - `Yolov5`: For object detction.
